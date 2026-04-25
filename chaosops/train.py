from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from env import ChaosOpsEnv
from wrapper import ChaosOpsWrapper

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.autograd.set_detect_anomaly(True)


@dataclass
class Transition:
    state_text: str
    action_text: str
    action_name: str
    reward: float
    sampled_logprob_scalar: float


@dataclass
class EpisodeTrace:
    success: bool
    total_reward: float
    steps: int
    transitions: List[Transition]


def _make_env(max_steps: int, seed: int) -> ChaosOpsEnv:
    random.seed(seed)
    torch.manual_seed(seed)
    return ChaosOpsEnv(max_steps=max_steps)


def _extract_observation(out: Dict[str, Any]) -> Dict[str, Any]:
    obs = out.get("observation")
    if isinstance(obs, dict):
        return obs
    st = out.get("state", {})
    if st.get("api_schema_version", 1) == 1:
        return {
            "service": "auth_service",
            "status": st.get("service_status", "crashed"),
            "cpu_limit": "500m",
        }
    return {
        "service_name": "auth_service",
        "condition": st.get("service_status", "crashed"),
        "max_compute": "1",
    }


def _reset_task3(env: ChaosOpsEnv) -> Dict[str, Any]:
    return env.reset("task3")


def load_4bit_qwen(model_name: str = "Qwen/Qwen2.5-0.5B") -> Tuple[Any, Any]:
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as unsloth_exc:
        print(f"[WARN] Unsloth unavailable: {unsloth_exc}")

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        return model, tokenizer
    except Exception as bnb_exc:
        raise RuntimeError(
            "4-bit model loading failed. Install Unsloth or bitsandbytes-enabled Transformers to continue."
        ) from bnb_exc


def load_unsloth_qwen(model_name: str = "Qwen/Qwen2.5-0.5B") -> Tuple[Any, Any, None]:
    """Backward-compatible loader used by eval.py."""
    model, tokenizer = load_4bit_qwen(model_name=model_name)
    return model, tokenizer, None


def phase_a_generate_action(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, float]:
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        generated = model.generate(
            enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    new_ids = generated.sequences[0, input_len:]
    action_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    sampled_logprob = 0.0
    if generated.scores:
        for step_idx, score_tensor in enumerate(generated.scores):
            if step_idx >= int(new_ids.shape[0]):
                break
            token_id = int(new_ids[step_idx].item())
            step_log_probs = torch.log_softmax(score_tensor[0], dim=-1)
            sampled_logprob = sampled_logprob + float(step_log_probs[token_id].item())

    del enc, generated, new_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return action_text[-1200:], sampled_logprob


def generate_model_action(
    model: Any,
    tokenizer: Any,
    fastlm: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 32,
) -> str:
    """Backward-compatible inference helper used by eval.py."""
    del fastlm
    action_text, _ = phase_a_generate_action(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=max(0.2, float(temperature)),
        top_p=0.9,
        max_new_tokens=min(32, int(max_new_tokens)),
    )
    return action_text


def safe_action_sampling(
    model: Any,
    tokenizer: Any,
    wrapper: ChaosOpsWrapper,
    observation: Dict[str, Any],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    epsilon_random: float,
) -> Tuple[str, Dict[str, Any], str, float]:
    prompt = wrapper.observation_to_prompt(observation)
    action_text, sampled_logprob = phase_a_generate_action(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    action_obj = wrapper.parse_model_output(action_text)
    if random.random() < epsilon_random:
        action_obj = {
            "action": random.choice(["query_system", "get_schema", "request_access", "fix_service"]),
            "payload": {},
        }
        action_text = json.dumps(action_obj, ensure_ascii=True)
    return prompt, action_obj, action_text, sampled_logprob


def phase_b_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    action_text: str,
) -> torch.Tensor:
    model.train()
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    action_enc = tokenizer(action_text, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=False)

    prompt_ids = prompt_enc["input_ids"].to(model.device)
    prompt_mask = prompt_enc.get("attention_mask")
    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt_ids)
    prompt_mask = prompt_mask.to(model.device)

    action_ids = action_enc["input_ids"].to(model.device)
    if action_ids.shape[1] == 0:
        return torch.zeros((), dtype=torch.float32, device=model.device, requires_grad=True)
    action_mask = torch.ones_like(action_ids, device=model.device)

    full_ids = torch.cat([prompt_ids, action_ids], dim=1)
    full_mask = torch.cat([prompt_mask, action_mask], dim=1)

    outputs = model(input_ids=full_ids, attention_mask=full_mask)
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = full_ids[:, 1:]

    prompt_len = int(prompt_ids.shape[1])
    start = max(0, prompt_len - 1)
    end = start + int(action_ids.shape[1])

    action_logits = shifted_logits[:, start:end, :]
    action_labels = shifted_labels[:, start:end]

    log_probs = torch.log_softmax(action_logits, dim=-1)
    selected_logprob = log_probs.gather(dim=-1, index=action_labels.unsqueeze(-1)).squeeze(-1)
    total_logprob = selected_logprob.sum()

    del prompt_enc, action_enc, prompt_ids, prompt_mask, action_ids, action_mask
    del full_ids, full_mask, outputs, logits, shifted_logits, shifted_labels
    del action_logits, action_labels, log_probs, selected_logprob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_logprob


def compute_shaped_reward(
    step_out: Dict[str, Any],
    prev_action: str,
    current_action: str,
) -> float:
    result = step_out.get("result", {})
    state = step_out.get("state", {})

    success = bool(state.get("service_status") == "running")
    useless_action = not bool(result.get("ok", False))
    repeated_action = prev_action == current_action and prev_action != ""
    system_worsened = bool(
        state.get("schema_fail_count", 0) > 0
        or result.get("error_code") in {"NO_PERMISSION", "WRONG_SCHEMA", "UNKNOWN_ACTION"}
    )

    reward = -0.01
    if success:
        reward = reward + 1.0
    if useless_action:
        reward = reward - 0.5
    if repeated_action:
        reward = reward - 0.2
    if system_worsened:
        reward = reward - 0.3
    return float(reward)


def discounted_rewards(rewards: List[float], gamma: float) -> List[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def collect_episode(
    env: ChaosOpsEnv,
    wrapper: ChaosOpsWrapper,
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    epsilon_random: float,
) -> EpisodeTrace:
    reset_out = _reset_task3(env)
    state = _extract_observation(reset_out)

    trajectory: List[Transition] = []
    prev_action = ""
    done = False
    last_step_out = reset_out

    for _ in range(env.max_steps):
        state_text = json.dumps(state, ensure_ascii=True)
        prompt, action_obj, action_text, sampled_logprob = safe_action_sampling(
            model=model,
            tokenizer=tokenizer,
            wrapper=wrapper,
            observation=state,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            epsilon_random=epsilon_random,
        )

        step_out = env.step(action_obj["action"], action_obj["payload"])
        reward = compute_shaped_reward(step_out, prev_action=prev_action, current_action=str(action_obj["action"]))

        trajectory.append(
            Transition(
                state_text=state_text,
                action_text=action_text,
                action_name=str(action_obj["action"]),
                reward=reward,
                sampled_logprob_scalar=float(sampled_logprob),
            )
        )

        state = _extract_observation(step_out)
        prev_action = str(action_obj["action"])
        done = bool(step_out.get("done", False))
        last_step_out = step_out

        del prompt, action_obj, action_text
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if done:
            break

    total_reward = float(sum(t.reward for t in trajectory))
    success = bool(last_step_out.get("state", {}).get("service_status") == "running")
    return EpisodeTrace(
        success=success,
        total_reward=total_reward,
        steps=len(trajectory),
        transitions=trajectory,
    )


def compute_policy_loss(
    model: Any,
    tokenizer: Any,
    wrapper: ChaosOpsWrapper,
    episodes: List[EpisodeTrace],
    gamma: float,
) -> Tuple[torch.Tensor, Counter[str], float]:
    all_returns: List[float] = []
    transition_list: List[Transition] = []
    action_counter: Counter[str] = Counter()

    for episode in episodes:
        rewards = [t.reward for t in episode.transitions]
        ep_returns = discounted_rewards(rewards, gamma=gamma)
        all_returns.extend(ep_returns)
        transition_list.extend(episode.transitions)
        for transition in episode.transitions:
            action_counter.update([transition.action_name])

    if not transition_list:
        raise RuntimeError("No transitions collected for policy update.")

    returns = torch.tensor(all_returns, dtype=torch.float32, device=model.device)
    baseline = returns.mean().detach()
    advantages = returns - baseline
    advantages = advantages.clone()
    advantages = advantages / (advantages.std(unbiased=False) + 1e-6)

    selected_logprobs: List[torch.Tensor] = []
    for transition in transition_list:
        state_dict = json.loads(transition.state_text)
        prompt = wrapper.observation_to_prompt(state_dict)
        lp = phase_b_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            action_text=transition.action_text,
        )
        if torch.isnan(lp) or torch.isinf(lp):
            raise RuntimeError("Invalid log probability encountered during training forward pass.")
        selected_logprobs.append(lp)

    selected_logprob_tensor = torch.stack(selected_logprobs)
    if not selected_logprob_tensor.requires_grad:
        raise RuntimeError("selected_logprob tensor does not require gradients.")

    loss = -((selected_logprob_tensor * advantages).mean())
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError("NaN/Inf loss detected before backward.")

    reward_variance = float(returns.var(unbiased=False).detach().item())
    return loss, action_counter, reward_variance


def train_loop(
    output_dir: str,
    model_name: str,
    train_steps: int,
    group_size: int,
    learning_rate: float,
    gamma: float,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_grad_norm: float,
    epsilon_random: float,
    variation_prob: float,
) -> Dict[str, Any]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if max_new_tokens > 32:
        raise ValueError("max_new_tokens must be <= 32 for memory safety.")
    if group_size > 2:
        raise ValueError("group_size must be <= 2 for memory safety.")

    model, tokenizer = load_4bit_qwen(model_name=model_name)
    env = _make_env(max_steps=10, seed=13)
    wrapper = ChaosOpsWrapper()
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logs: List[Dict[str, Any]] = []
    recent_losses: List[float] = []
    recent_avg_rewards: List[float] = []

    for update in range(train_steps):
        episodes: List[EpisodeTrace] = []
        for _ in range(group_size):
            episode = collect_episode(
                env=env,
                wrapper=wrapper,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                epsilon_random=min(0.5, max(0.0, epsilon_random + 0.5 * variation_prob)),
            )
            episodes.append(episode)

        try:
            loss, action_counter, reward_variance = compute_policy_loss(
                model=model,
                tokenizer=tokenizer,
                wrapper=wrapper,
                episodes=episodes,
                gamma=gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        except RuntimeError as exc:
            print(f"[AUTOGRAD_ERROR] {str(exc)}")
            raise

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            raise RuntimeError("Invalid gradient norm detected.")
        optimizer.step()

        avg_reward = float(sum(ep.total_reward for ep in episodes) / max(1, len(episodes)))
        success_rate = float(sum(1 for ep in episodes if ep.success) / max(1, len(episodes)))
        loss_value = float(loss.detach().item())

        total_actions = max(1, sum(action_counter.values()))
        top_action_count = max(action_counter.values()) if action_counter else 0
        repeat_ratio = float(top_action_count / total_actions)

        if repeat_ratio > 0.70 and update >= 3:
            raise RuntimeError(f"Action diversity check failed. Repetition ratio={repeat_ratio:.3f}")
        if reward_variance <= 1e-8 and update >= 2:
            raise RuntimeError("Reward variance is too low; rewards appear constant.")

        log_row = {
            "update": update,
            "avg_reward": avg_reward,
            "reward_variance": reward_variance,
            "success_rate": success_rate,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "action_distribution": dict(action_counter),
        }
        logs.append(log_row)
        print(json.dumps(log_row, ensure_ascii=True))

        if episodes and episodes[0].transitions:
            episode_debug = [
                {
                    "state": tr.state_text,
                    "action": tr.action_name,
                    "reward": tr.reward,
                }
                for tr in episodes[0].transitions
            ]
            print(json.dumps({"episode_trace": episode_debug}, ensure_ascii=True))

        recent_losses.append(loss_value)
        recent_avg_rewards.append(avg_reward)
        if len(recent_losses) >= 5:
            if max(recent_losses[-5:]) - min(recent_losses[-5:]) <= 1e-8:
                raise RuntimeError("Loss did not change across recent updates.")
        if len(recent_avg_rewards) >= 5:
            if max(recent_avg_rewards[-5:]) - min(recent_avg_rewards[-5:]) <= 1e-8:
                raise RuntimeError("Average reward did not change across recent updates.")

        del episodes, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "completed_steps": len(logs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./chaosops-qwen-grpo")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epsilon_random", type=float, default=0.20)
    parser.add_argument("--variation_prob", type=float, default=0.30)
    args = parser.parse_args()

    train_steps = args.episodes if args.episodes is not None else args.train_steps
    result = train_loop(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_steps=train_steps,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_grad_norm=args.max_grad_norm,
        epsilon_random=args.epsilon_random,
        variation_prob=args.variation_prob,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
