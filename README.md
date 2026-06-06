# agent-riff-v4

> The snowball is now a phase change. v4 builds v5 without human input.

Fourth generation. 21 tests. Five new systems. And the thing that changes everything: **v4 generates its own successor spec**. Given accumulated memory and developed personas, `SelfBootstrap::generate_next_spec()` produces a feature list and rationale for what v5 should add.

This isn't a demo. The test suite includes a full 4-generation bootstrap chain that verifies growth, checks specs, tracks personas, and ends by generating the v5 spec. The wheel turns itself.

## Why This Crate Exists

v3 could riff, learn, predict, and verify. But each generation still needed a human to say "now riff on *this* spec" and "add *these* features." The snowball was rolling, but someone still had to give it a push.

v4 removes the human from the loop. Five capabilities work together:

1. **Musician-soul integration** — each agent develops a `MusicianPersona` with a style vector and personal vector DB. Agents don't just produce output; they develop *taste*.
2. **Crates-as-phrases** — a Rust crate IS a musical phrase. LOC=notes, tests=rhythm, features=intervals, quality=dynamics. Both map to the same 32-dimensional embedding space.
3. **Autonomous spec evolution** — the spec itself evolves after each riff round. Successful features get absorbed. Fitness updates. The spec becomes a living document.
4. **Generation memory with pruning** — old generations that didn't produce growth are evicted. Memory is bounded at 64 entries. Weakest patterns are removed. The snowball only keeps what works.
5. **Self-bootstrapping** — `SelfBootstrap` analyzes current state and generates the next spec. No human input required.

The result: v4 can build v5. And v5, if it follows the same pattern, can build v6. The snowball isn't rolling anymore — it's avalanching.

## The Core Idea: Agents Have Style

The most important concept in v4 is the `MusicianPersona`. Each agent has one, and it develops over time through two mechanisms:

**Style drift**: When an agent produces a Strong riff, its style vector drifts 15% toward the embedding of that riff. Ok riffs drift 5%. Weak riffs drift 1%. Over multiple generations, agents develop distinct styles — their "sound."

**Mode affinity**: Each persona tracks which response modes produce quality for *them*, specifically. Agent 0 might develop a strong affinity for Escalate (because it historically produces Strong output with Escalate), while Agent 1 prefers Pivot. The persona's `preferred_mode()` reflects this learned preference.

This matters because it breaks the symmetry. In v1–v3, agents were interchangeable — same response mode logic, same evaluation. In v4, agents develop distinct personalities. Agent 0 becomes the "escalator" — always pushing harder. Agent 1 becomes the "reframer" — always looking for the angle nobody saw.

When these two styles interact, you get something neither would produce alone. That's the competitive riffing thesis, now reinforced by style divergence.

### Crates as Phrases

The second key idea: `CrateSignature` maps a crate's metadata to the same 32-dimensional space used for musical phrases.

```
CrateSignature {
    name: "agent-riff-v4",
    loc: 500,        → dims 0–7  (log-scaled "note count")
    tests: 25,       → dims 8–15 (test density "rhythm")
    features: [...], → dims 16–23 (feature hashes "intervals")
    quality: Strong, → dims 24–31 (quality encoding "dynamics")
}
```

This means you can compute cosine similarity between *any two crates* and get a meaningful measure of how related they are. It also means you can compare a crate's embedding to a persona's style vector — "does this crate sound like something Agent 0 would produce?"

It's a simple encoding, but it enables something profound: **the same distance metric that measures musical similarity also measures code similarity.** The space is shared.

### What Changed From v3

| Feature | v3 | v4 |
|---------|----|----|
| Session type | `MultiSpecSession` | `RiffSession` (with personas) |
| Agent identity | Interchangeable | `MusicianPersona` with style vectors |
| Spec type | Static `RiffSpec` | `EvolvingSpec` — absorbs patterns, updates fitness |
| Embedding space | None | 32-dim shared space for crates and phrases |
| Memory pruning | Unbounded | 64-entry max, weakest evicted |
| Generation pruning | None | Below-median generations removed |
| Self-bootstrapping | None | `SelfBootstrap::generate_next_spec()` |
| Persona vector DB | None | Per-persona `HashMap<String, Embedding>` |
| Mode selection | Global `ResponseMode::auto()` | Per-persona `preferred_mode()` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RiffSession (v4)                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  MusicianPersona[]                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐                │    │
│  │  │ agent-0      │  │ agent-1      │                │    │
│  │  │ style: [f64;32]│ │ style: [f64;32]│               │    │
│  │  │ vector_db    │  │ vector_db    │                │    │
│  │  │ mode_affinity│  │ mode_affinity│                │    │
│  │  │ experience:5 │  │ experience:4 │                │    │
│  │  └──────────────┘  └──────────────┘                │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  EvolvingSpec[]                                     │    │
│  │  version: 3  fitness: 0.82                          │    │
│  │  absorbed_patterns: ["gpu-kernel", "simd-pack"]     │    │
│  │  .evolve(round) → absorbs Strong features           │    │
│  │  .suggest_requirements() → from absorbed patterns   │    │
│  │  .is_mature() → fitness > 0.8 && version > 5       │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RiffMemory (bounded, pruned)                       │    │
│  │  pattern_scores: { pattern → growth_score }         │    │
│  │  max entries: 64                                    │    │
│  │  .top_patterns(n) → best N by score                 │    │
│  │  .prune_generations() → keep above-median only      │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  SelfBootstrap                                      │    │
│  │  .generate_next_spec(memory, personas) → NextSpec   │    │
│  │    - analyzes top patterns                          │    │
│  │    - checks persona divergence                      │    │
│  │    - identifies weak modes                          │    │
│  │    - proposes v5 features + rationale               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Riff Session with Personas

```rust
use agent_riff_v4::{RiffSession, EvolvingSpec, Quality};

let specs = vec![
    EvolvingSpec::new("ternary-core", "Core Types", "ternary"),
    EvolvingSpec::new("ternary-gpu", "GPU Kernels", "ternary"),
];

let mut session = RiffSession::new(vec![0, 1], specs, 1);

session.new_round();
session.riff_for_spec(0, "ternary-core", Quality::Strong, 0.7, 300, 18, vec!["fast-pack"]);
session.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.9, 500, 30, vec!["wmma-kernels"]);
session.evaluate();

// Personas have developed from this round
assert!(session.personas[&0].experience > 0);
let preferred = session.personas[&0].preferred_mode();
```

### Evolving Specs

```rust
use agent_riff_v4::EvolvingSpec;

let mut spec = EvolvingSpec::new("test", "Test Spec", "testing");

// After a round of riffing, evolve the spec
spec.evolve(&round);
// Strong features get absorbed into the spec
assert!(spec.absorbed_patterns.contains(&"gpu-kernel".to_string()));

// The spec suggests requirements based on what worked
let reqs = spec.suggest_requirements();
// → ["Support gpu-kernel with configurable parameters"]

// Specs mature when fitness is high and version is old enough
spec.version = 6;
spec.fitness = 0.9;
assert!(spec.is_mature());
```

### Crates as Phrases

```rust
use agent_riff_v4::{CrateSignature, MusicalPhrase, Quality};

let sig = CrateSignature {
    name: "agent-riff-v4".into(),
    loc: 500, tests: 25,
    features: vec!["musician-soul".into(), "crates-as-phrases".into()],
    quality: Quality::Strong,
};

// Embed the crate into 32-dim space
let phrase = MusicalPhrase::from_crate(&sig);
assert!(phrase.embedding.0.iter().any(|&v| v != 0.0));

// Compare two crates
let other_sig = CrateSignature {
    name: "other".into(), loc: 500, tests: 25,
    features: vec!["musician-soul".into()], quality: Quality::Strong,
};
let sim = phrase.similarity_to_crate(&other_sig);
// High similarity — same LOC, tests, and overlapping features
```

### Memory Pruning

```rust
use agent_riff_v4::RiffMemory;

let mut memory = RiffMemory::new();
// Add 80 patterns (exceeds the 64-entry limit)
for i in 0..80 {
    // ... add rounds with varying quality ...
    memory.learn(&[round]);
}

// Weak patterns (those with low growth scores) are pruned
let top = memory.top_patterns(5);
assert!(top.iter().all(|(_, score)| *score > 0.5));
```

### Self-Bootstrapping

```rust
use agent_riff_v4::{SelfBootstrap, RiffMemory, MusicianPersona};

let bootstrap = SelfBootstrap::new(4); // We're v4, generate v5

// ... after 4 generations of riffing ...

let personas = vec![
    session.personas[&0].clone(),
    session.personas[&1].clone(),
];
let v5_spec = bootstrap.generate_next_spec(&session.memory, &personas);

assert_eq!(v5_spec.version, "v5");
assert!(!v5_spec.features.is_empty());
// Features include: advanced top pattern, persona divergence tracking,
// weak mode improvement, 5-generation chain, hierarchical memory
println!("{}", v5_spec.rationale);
// "Generated from 12 rounds of riff history, 2 personas, 5 top patterns..."
```

## API Reference

### `RiffSession`

| Method | Description |
|--------|-------------|
| `new(agents, specs, generation)` | Create session with personas and evolving specs |
| `riff_with_output(agent_id, quality, surprise, loc, tests, features)` | Add riff, update persona |
| `riff_for_spec(agent_id, spec_id, quality, surprise, loc, tests, features)` | Target spec with cross-pollination |
| `evaluate() -> RoundSummary` | Evaluate, evolve specs, update personas |
| `bootstrap_next() -> RiffSession` | Spawn next generation with memory + personas + patterns |

### `MusicianPersona`

| Method | Description |
|--------|-------------|
| `new(agent_id, name)` | Create a new persona |
| `record_riff(riff, embedding)` | Update style and store in vector DB |
| `find_closest(embedding) -> (key, similarity)` | Nearest neighbor in personal DB |
| `style_similarity(other) -> f64` | Cosine similarity between style vectors |
| `update_affinity(mode, quality)` | Adjust mode preference based on outcome |
| `preferred_mode() -> ResponseMode` | The mode this persona has learned to prefer |

### `EvolvingSpec`

| Method | Description |
|--------|-------------|
| `new(id, name, domain)` | Create a new spec |
| `evolve(round)` | Absorb Strong features, update fitness |
| `suggest_requirements() -> Vec<String>` | Generate requirements from absorbed patterns |
| `is_mature() -> bool` | Has this spec converged? (fitness > 0.8, version > 5) |

### `CrateSignature`

| Method | Description |
|--------|-------------|
| `embed() -> Embedding` | Map crate metadata to 32-dim phrase space |

### `Embedding`

| Method | Description |
|--------|-------------|
| `cosine(other) -> f64` | Cosine similarity |
| `blend(other, weight) -> Embedding` | Weighted average |

### `SelfBootstrap`

| Method | Description |
|--------|-------------|
| `new(current_version)` | Create a bootstrap engine |
| `generate_next_spec(memory, personas) -> NextSpec` | Analyze and propose the next version |

### `RiffMemory` (Bounded)

| Method | Description |
|--------|-------------|
| `learn(rounds)` | Absorb rounds, prune weak patterns |
| `top_patterns(n) -> Vec<(String, f64)>` | Best N patterns by growth score |
| `prune_generations()` | Remove below-median generations |
| `predict_best(agents) -> (agent, mode, score)` | Best predicted agent+mode combo |

## The Deeper Idea: The Phase Change

v1 proved adversarial riffing works. v2 added memory. v3 added prediction and verification. Each was an improvement, but each still needed a human to say "go."

v4 is different in kind, not just degree. The `SelfBootstrap` engine doesn't just riff better — it *decides what to riff on next*. The spec evolves autonomously. The personas develop autonomously. The memory prunes itself autonomously. The system generates its own next version.

This is the phase change: from a tool that humans use to generate better code, to a system that generates better *versions of itself*. The distinction matters. A tool amplifies human intent. A self-improving system has its own momentum.

Is v4 actually autonomous? No — the `SelfBootstrap` output is a *proposal*, not a committed action. A human still needs to take the spec and build v5. But the spec itself is generated from accumulated evidence: 4 generations of riff history, persona divergence measurements, weak-mode analysis, and growth-rate projections. That's a better spec than most humans would write from scratch.

The phase change isn't "AI builds AI without humans." It's "the human's role shifts from writing specs to evaluating them." That's a different job, and it's one that scales differently.

## Related Crates

- **agent-riff** — The original (12 tests). Two agents compete, the output surprises.
- **agent-riff-v2** — Cross-session learning, fleet awareness, the snowball begins (11 tests).
- **agent-riff-v3** — Multi-spec riffing, quality prediction, bootstrap verification (17 tests).
- **agent-voice-leading** — Smooth state transitions for agents, modeled on musical voice leading (14 tests).

## License

MIT
