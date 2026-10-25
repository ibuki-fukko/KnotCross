"""Small NumPy neural-network toolkit for the NURBS intersection study."""

from __future__ import annotations

import numpy as np


class Dense:
    def __init__(self, input_dim: int, output_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / max(input_dim, 1))
        self.weight = rng.normal(0.0, scale, size=(input_dim, output_dim))
        self.bias = np.zeros(output_dim, dtype=np.float64)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self.cache: list[np.ndarray] = []
        self.m_weight = np.zeros_like(self.weight)
        self.v_weight = np.zeros_like(self.weight)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.cache.append(inputs)
        return inputs @ self.weight + self.bias

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if not self.cache:
            raise RuntimeError("Dense.backward called before forward")
        inputs = self.cache.pop()
        self.grad_weight += inputs.T @ gradient
        self.grad_bias += np.sum(gradient, axis=0)
        return gradient @ self.weight.T

    def zero_grad(self) -> None:
        self.grad_weight.fill(0.0)
        self.grad_bias.fill(0.0)

    def clear_cache(self) -> None:
        self.cache.clear()

    def step(self, learning_rate: float, beta1: float, beta2: float, epsilon: float, step_index: int) -> None:
        self.m_weight = beta1 * self.m_weight + (1.0 - beta1) * self.grad_weight
        self.v_weight = beta2 * self.v_weight + (1.0 - beta2) * (self.grad_weight * self.grad_weight)
        self.m_bias = beta1 * self.m_bias + (1.0 - beta1) * self.grad_bias
        self.v_bias = beta2 * self.v_bias + (1.0 - beta2) * (self.grad_bias * self.grad_bias)
        correction_one = 1.0 - beta1**step_index
        correction_two = 1.0 - beta2**step_index
        weight_update = (self.m_weight / correction_one) / (np.sqrt(self.v_weight / correction_two) + epsilon)
        bias_update = (self.m_bias / correction_one) / (np.sqrt(self.v_bias / correction_two) + epsilon)
        self.weight -= learning_rate * weight_update
        self.bias -= learning_rate * bias_update


class ReLU:
    def __init__(self):
        self.cache: list[np.ndarray] = []

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.cache.append(inputs > 0.0)
        return np.maximum(inputs, 0.0)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if not self.cache:
            raise RuntimeError("ReLU.backward called before forward")
        return gradient * self.cache.pop()

    def zero_grad(self) -> None:
        return None

    def clear_cache(self) -> None:
        self.cache.clear()

    def step(self, learning_rate: float, beta1: float, beta2: float, epsilon: float, step_index: int) -> None:
        return None


class Sequential:
    def __init__(self, layers: list[Dense | ReLU]):
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        caches = []
        value = inputs
        for layer in self.layers:
            value = layer.forward(value)
            caches.append(value)
        return value, caches

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        value = gradient
        for layer in reversed(self.layers):
            value = layer.backward(value)
        return value

    def parameter_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()

    def step(self, learning_rate: float, beta1: float, beta2: float, epsilon: float, step_index: int) -> None:
        for layer in self.layers:
            layer.step(learning_rate, beta1, beta2, epsilon, step_index)

    def state(self, prefix: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                result[f"{prefix}.dense{dense_index}.weight"] = layer.weight.copy()
                result[f"{prefix}.dense{dense_index}.bias"] = layer.bias.copy()
                dense_index += 1
        return result

    def load_state(self, state: dict[str, np.ndarray], prefix: str) -> None:
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weight[...] = state[f"{prefix}.dense{dense_index}.weight"]
                layer.bias[...] = state[f"{prefix}.dense{dense_index}.bias"]
                dense_index += 1


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, rng: np.random.Generator) -> Sequential:
    layers: list[Dense | ReLU] = []
    dimensions = [input_dim] + hidden_dims + [output_dim]
    for index in range(len(dimensions) - 1):
        layers.append(Dense(dimensions[index], dimensions[index + 1], rng))
        if index < len(dimensions) - 2:
            layers.append(ReLU())
    return Sequential(layers)


class SharedResidualIntersectionModel:
    """Shared curve encoder plus residual-field fusion and two output heads."""

    def __init__(self, curve_input_dim: int, residual_dim: int, map_dim: int, max_slots: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.encoder = make_mlp(curve_input_dim, [128, 128, 64], 64, rng)
        self.fusion = make_mlp(64 + 64 + residual_dim, [128, 64], 64, rng)
        self.map_head = make_mlp(64, [64], map_dim, rng)
        self.slot_head = make_mlp(64, [64], max_slots * 3, rng)
        self.max_slots = max_slots
        self.map_dim = map_dim
        self.state_cache: dict[str, object] = {}

    def forward(self, curve_a: np.ndarray, curve_b: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.clear_cache()
        feature_a, cache_a = self.encoder.forward(curve_a)
        feature_b, cache_b = self.encoder.forward(curve_b)
        fused_input = np.concatenate([feature_a, feature_b, residual], axis=1)
        fused, fusion_cache = self.fusion.forward(fused_input)
        map_logits, map_cache = self.map_head.forward(fused)
        slot_logits, slot_cache = self.slot_head.forward(fused)
        self.state_cache = {
            "cache_a": cache_a,
            "cache_b": cache_b,
            "fusion_cache": fusion_cache,
            "map_cache": map_cache,
            "slot_cache": slot_cache,
            "fused_input": fused_input,
        }
        return map_logits, slot_logits

    def backward(self, map_gradient: np.ndarray, slot_gradient: np.ndarray) -> None:
        gradient_fused_from_map = self.map_head.backward(map_gradient)
        gradient_fused_from_slots = self.slot_head.backward(slot_gradient)
        gradient_fused = gradient_fused_from_map + gradient_fused_from_slots
        gradient_fused_input = self.fusion.backward(gradient_fused)
        feature_dim = 64
        gradient_a = gradient_fused_input[:, :feature_dim]
        gradient_b = gradient_fused_input[:, feature_dim : 2 * feature_dim]
        self.encoder.backward(gradient_b)
        self.encoder.backward(gradient_a)

    def clear_cache(self) -> None:
        self.encoder.clear_cache()
        self.fusion.clear_cache()
        self.map_head.clear_cache()
        self.slot_head.clear_cache()

    def parameter_layers(self) -> list[Dense]:
        return (
            self.encoder.parameter_layers()
            + self.fusion.parameter_layers()
            + self.map_head.parameter_layers()
            + self.slot_head.parameter_layers()
        )

    def zero_grad(self) -> None:
        for layer in self.parameter_layers():
            layer.zero_grad()

    def step(self, learning_rate: float, step_index: int) -> None:
        for layer in self.parameter_layers():
            layer.step(learning_rate, 0.9, 0.999, 1e-8, step_index)

    def state(self) -> dict[str, np.ndarray]:
        state = {}
        state.update(self.encoder.state("encoder"))
        state.update(self.fusion.state("fusion"))
        state.update(self.map_head.state("map_head"))
        state.update(self.slot_head.state("slot_head"))
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.encoder.load_state(state, "encoder")
        self.fusion.load_state(state, "fusion")
        self.map_head.load_state(state, "map_head")
        self.slot_head.load_state(state, "slot_head")


class DirectIntersectionModel:
    """Ablation that bypasses the shared curve encoder."""

    def __init__(self, input_dim: int, map_dim: int, max_slots: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.backbone = make_mlp(input_dim, [128, 64], 64, rng)
        self.map_head = make_mlp(64, [64], map_dim, rng)
        self.slot_head = make_mlp(64, [64], max_slots * 3, rng)
        self.state_cache: dict[str, object] = {}
        self.max_slots = max_slots
        self.map_dim = map_dim

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.clear_cache()
        fused, backbone_cache = self.backbone.forward(features)
        map_logits, map_cache = self.map_head.forward(fused)
        slot_logits, slot_cache = self.slot_head.forward(fused)
        self.state_cache = {"backbone_cache": backbone_cache, "map_cache": map_cache, "slot_cache": slot_cache}
        return map_logits, slot_logits

    def clear_cache(self) -> None:
        self.backbone.clear_cache()
        self.map_head.clear_cache()
        self.slot_head.clear_cache()

    def backward(self, map_gradient: np.ndarray, slot_gradient: np.ndarray) -> None:
        gradient = self.map_head.backward(map_gradient) + self.slot_head.backward(slot_gradient)
        self.backbone.backward(gradient)

    def parameter_layers(self) -> list[Dense]:
        return self.backbone.parameter_layers() + self.map_head.parameter_layers() + self.slot_head.parameter_layers()

    def zero_grad(self) -> None:
        for layer in self.parameter_layers():
            layer.zero_grad()

    def step(self, learning_rate: float, step_index: int) -> None:
        for layer in self.parameter_layers():
            layer.step(learning_rate, 0.9, 0.999, 1e-8, step_index)

    def state(self) -> dict[str, np.ndarray]:
        state = {}
        state.update(self.backbone.state("backbone"))
        state.update(self.map_head.state("map_head"))
        state.update(self.slot_head.state("slot_head"))
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.backbone.load_state(state, "backbone")
        self.map_head.load_state(state, "map_head")
        self.slot_head.load_state(state, "slot_head")


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))

