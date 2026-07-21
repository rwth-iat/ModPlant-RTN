"""Runtime-control variants derived from the seeded RTN General Recipe."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

from random_recipe import generate_random_recipe_spec
from recipe_graph_adapter import recipe_ir_to_graph
from recipe_ir import auto_enrich_recipe_spec, build_recipe_ir
from modplant_recipe import RecipeEdge, RecipeGraph, RecipeNode, export_general_recipe


def seeded_base_graph(
    seed: int = 42,
    *,
    enriched: bool = True,
    ingredient_count: int = 3,
) -> RecipeGraph:
    """Return the existing notebook's nonlinear General Recipe graph."""
    spec = generate_random_recipe_spec(
        seed,
        topology="linear",
        ingredient_count=ingredient_count,
    )
    if enriched:
        spec = auto_enrich_recipe_spec(spec)
    graph = recipe_ir_to_graph(build_recipe_ir(spec))
    graph.metadata.update({"runtimeVariant": "base", "derivedFromSeed": seed})
    return graph


def or_variant(
    seed: int = 42,
    *,
    enriched: bool = True,
    ingredient_count: int = 3,
) -> RecipeGraph:
    """Add an optimizer-selected inclusive branch set after mixing."""
    spec = generate_random_recipe_spec(
        seed,
        topology="linear",
        ingredient_count=ingredient_count,
    )
    if enriched:
        spec = auto_enrich_recipe_spec(spec)
    procedure = list(spec["procedure"])
    mix_index = next(index for index, step in enumerate(procedure) if "mix" in step)
    procedure.insert(
        mix_index + 1,
        {
            "inclusive": [
                {"branch_id": "sample", "steps": [{"usage": {"duration_s": 45}}]},
                {"branch_id": "analyze", "steps": [{"usage": {"duration_s": 75}}]},
                {"branch_id": "condition", "steps": [{"usage": {"duration_s": 105}}]},
            ],
            "min_branches": 2,
            "max_branches": 2,
        },
    )
    spec["procedure"] = procedure
    spec["id"] = f"RTN_Random_nonlinear_OR_{seed}"
    spec["metadata"] = {**spec.get("metadata", {}), "runtimeVariant": "or"}
    return recipe_ir_to_graph(build_recipe_ir(spec))


def runtime_loop_variant(
    seed: int = 42,
    *,
    enriched: bool = True,
    ingredient_count: int = 3,
) -> RecipeGraph:
    """Add a runtime quality loop with a finite three-iteration planning envelope."""
    graph = deepcopy(
        seeded_base_graph(
            seed,
            enriched=enriched,
            ingredient_count=ingredient_count,
        )
    )
    graph.id = f"RTN_Random_nonlinear_RuntimeLoop_{seed}"
    graph.metadata["runtimeVariant"] = "runtime-loop"
    mix = next(node for node in graph.nodes if node.activity_type == "mix")
    original = next(edge for edge in graph.outgoing(mix.id) if edge.flow_type == "Control")
    graph.edges.remove(original)
    loop_id = "quality_conditioning_loop"
    body = RecipeGraph(
        id=f"{graph.id}_LoopBody",
        recipe_level="General",
        nodes=[
            RecipeNode("loop_start", "Start", "Loop start"),
            RecipeNode(
                "conditioning_mix",
                "Activity",
                "Quality conditioning mix",
                "mix",
                parameters={"rpm": 100, "duration_s": 20},
                metadata={"processElementType": "Process Operation"},
            ),
            RecipeNode("loop_end", "End", "Loop end"),
        ],
        edges=[
            RecipeEdge("loop_e1", "loop_start", "conditioning_mix"),
            RecipeEdge("loop_e2", "conditioning_mix", "loop_end"),
        ],
    )
    graph.nodes.append(
        RecipeNode(
            loop_id,
            "LoopRegion",
            "Condition until quality is accepted",
            loop={
                "minIterations": 1,
                "maxIterations": 3,
                "planningIterations": 3,
                "exitCondition": {"op": "ref", "name": "quality.accepted"},
                "body": body.to_dict(),
            },
            metadata={"runtimeDecision": True},
        )
    )
    graph.edges.extend(
        [
            RecipeEdge("runtime_loop_in", mix.id, loop_id),
            RecipeEdge("runtime_loop_out", loop_id, original.target),
        ]
    )
    return graph


def conditional_jump_variant(
    seed: int = 42,
    *,
    enriched: bool = True,
    ingredient_count: int = 3,
) -> RecipeGraph:
    """Add a forward conditional jump that skips usage when quality permits."""
    graph = deepcopy(
        seeded_base_graph(
            seed,
            enriched=enriched,
            ingredient_count=ingredient_count,
        )
    )
    graph.id = f"RTN_Random_nonlinear_ConditionalJump_{seed}"
    graph.metadata["runtimeVariant"] = "conditional-jump"
    mix = next(node for node in graph.nodes if node.activity_type == "mix")
    settling = next(node for node in graph.nodes if node.activity_type == "settling")
    graph.edges.append(
        RecipeEdge(
            "jump_skip_usage",
            mix.id,
            settling.id,
            flow_type="Jump",
            condition={"op": "ref", "name": "quality.skipUsage"},
            priority=1,
            metadata={"runtimeDecision": True},
        )
    )
    return graph


def combined_runtime_variant(
    seed: int = 42,
    *,
    enriched: bool = True,
    ingredient_count: int = 3,
) -> RecipeGraph:
    """Combine OR, runtime loop, and a conditional forward jump in one recipe."""
    graph = deepcopy(
        or_variant(
            seed,
            enriched=enriched,
            ingredient_count=ingredient_count,
        )
    )
    graph.id = f"RTN_Random_nonlinear_OR_Loop_Jump_{seed}"
    graph.metadata["runtimeVariant"] = "or-loop-jump"

    mix = next(node for node in graph.nodes if node.activity_type == "mix")
    original = next(edge for edge in graph.outgoing(mix.id) if edge.flow_type == "Control")
    graph.edges.remove(original)
    loop_graph = runtime_loop_variant(
        seed,
        enriched=enriched,
        ingredient_count=ingredient_count,
    )
    loop_node = next(node for node in loop_graph.nodes if node.kind == "LoopRegion")
    graph.nodes.append(deepcopy(loop_node))
    graph.edges.extend(
        [
            RecipeEdge("combined_loop_in", mix.id, loop_node.id),
            RecipeEdge("combined_loop_out", loop_node.id, original.target),
        ]
    )
    settling = next(node for node in graph.nodes if node.activity_type == "settling")
    graph.edges.append(
        RecipeEdge(
            "combined_jump_skip_optional_work",
            loop_node.id,
            settling.id,
            flow_type="Jump",
            condition={"op": "ref", "name": "quality.skipOptionalWork"},
            priority=1,
            metadata={"runtimeDecision": True},
        )
    )
    return graph


def write_runtime_variant_general_recipes(
    destination: str | Path,
    seed: int = 42,
) -> Dict[str, str]:
    """Write deterministic BatchML General Recipe fixtures for all variants."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "or": or_variant(seed),
        "runtime-loop": runtime_loop_variant(seed),
        "conditional-jump": conditional_jump_variant(seed),
        "combined": combined_runtime_variant(seed),
    }
    paths: Dict[str, str] = {}
    for name, graph in variants.items():
        path = output / f"GeneralRecipe_{graph.id}.xml"
        export_general_recipe(graph, path, mode="annotated")
        paths[name] = str(path)
    return paths
