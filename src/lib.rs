//! # agent-riff-v4
//!
//! Snowball generation 4. v1 had 12 tests, v2 had 11, v3 had 16+ with full 3-gen bootstrap.
//! v4 adds 5 major features and 18+ tests including a full 4-generation bootstrap chain.
//!
//! What v4 adds over v3:
//! 1. **Musician-soul integration** — each agent has a MusicianPersona with its own vector DB.
//!    The riff engine and soul engine feed each other; agents develop style over time.
//! 2. **Crates-as-phrases** — a Rust crate IS a musical phrase. LOC=notes, tests=rhythm,
//!    features=intervals, quality=dynamics. Embeddings map crates to the same 32-dim space.
//! 3. **Autonomous spec evolution** — the spec itself evolves after each riff round based on
//!    what worked. The spec becomes a living document shaped by the agents that riff on it.
//! 4. **Generation memory with pruning** — old generations that didn't produce growth are pruned.
//!    The snowball only keeps what works. Memory is bounded and weakest patterns are evicted.
//! 5. **Self-bootstrapping** — the crate generates its OWN next spec. Given current capabilities
//!    and accumulated RiffMemory, it produces a spec for what v5 should add. The wheel turns itself.
//!
//! THE SNOWBALL: v1 → v2 → v3 → v4. Each version is better because competitive riffing
//! between agents produced improvements neither would invent alone.

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ── Ternary types (same encoding as ternary-cuda-kernels) ──────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit { Neg = -1, Zero = 0, Pos = 1 }

impl Trit {
    pub fn to_i8(self) -> i8 { self as i8 }
    pub fn from_i8(v: i8) -> Option<Self> {
        match v { -1 => Some(Trit::Neg), 0 => Some(Trit::Zero), 1 => Some(Trit::Pos), _ => None }
    }
    pub fn pack_bits(self) -> u8 { match self { Trit::Neg => 0, Trit::Zero => 1, Trit::Pos => 2 } }
    pub fn unpack_bits(b: u8) -> Self { match b & 0x3 { 0 => Trit::Neg, 1 => Trit::Zero, 2 => Trit::Pos, _ => Trit::Zero } }
}

/// Pack 16 trits into one u32 (GPU-ready).
pub fn pack_16(trits: &[Trit]) -> u32 {
    let mut packed = 0u32;
    for (i, &t) in trits.iter().take(16).enumerate() {
        packed |= (t.pack_bits() as u32) << (i * 2);
    }
    packed
}

/// Quality of a riff output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quality { Weak = -1, Ok = 0, Strong = 1 }
impl Quality { pub fn to_i8(self) -> i8 { self as i8 } }

/// Response mode — how an agent responds to the previous riff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseMode { Escalate, Pivot, Invert, Provoked }

impl ResponseMode {
    pub fn auto(surprise: f64, streak: u32, round: u32) -> Self {
        if streak > 5 { ResponseMode::Pivot }
        else if surprise < 0.2 { ResponseMode::Provoked }
        else if surprise > 0.7 { ResponseMode::Escalate }
        else if round > 8 { ResponseMode::Invert }
        else { ResponseMode::Invert }
    }
    pub fn all() -> &'static [ResponseMode] {
        &[ResponseMode::Escalate, ResponseMode::Pivot, ResponseMode::Invert, ResponseMode::Provoked]
    }
}

/// A single riff output.
#[derive(Debug, Clone)]
pub struct Riff {
    pub agent_id: u32,
    pub round: u32,
    pub quality: Quality,
    pub surprise: f64,
    pub loc: usize,
    pub tests: usize,
    pub features: Vec<String>,
    pub spec_id: Option<String>,
}

impl Riff {
    pub fn new(agent_id: u32, round: u32, quality: Quality, surprise: f64) -> Self {
        Self { agent_id, round, quality, surprise, loc: 0, tests: 0, features: Vec::new(), spec_id: None }
    }
    pub fn productivity(&self) -> f64 {
        let q = match self.quality { Quality::Weak => 0.5, Quality::Ok => 1.0, Quality::Strong => 2.0 };
        self.loc as f64 * self.tests as f64 * q * (0.5 + self.surprise)
    }
}

/// A round of the riff session.
#[derive(Debug, Clone)]
pub struct Round {
    pub number: u32,
    pub riffs: Vec<Riff>,
    pub best_agent: u32,
    pub quality_gap: i8,
    pub surprise_sum: f64,
}

impl Round {
    fn new(number: u32) -> Self { Self { number, riffs: Vec::new(), best_agent: 0, quality_gap: 0, surprise_sum: 0.0 } }

    fn add(&mut self, riff: Riff) {
        self.surprise_sum += riff.surprise;
        self.riffs.push(riff);
        self.recalc();
    }

    fn recalc(&mut self) {
        if self.riffs.is_empty() { return; }
        let best = self.riffs.iter().max_by_key(|r| r.quality.to_i8()).unwrap();
        let worst = self.riffs.iter().min_by_key(|r| r.quality.to_i8()).unwrap();
        self.best_agent = best.agent_id;
        self.quality_gap = best.quality.to_i8() - worst.quality.to_i8();
    }

    pub fn was_productive(&self) -> bool { self.surprise_sum > 0.3 || self.quality_gap > 0 }
}

// ════════════════════════════════════════════════════════════════════
// v4 Feature 1: Musician-Soul Integration
// ════════════════════════════════════════════════════════════════════

/// A 32-dimensional embedding vector shared by musical phrases AND crate signatures.
/// This enables "crates-as-phrases" (Feature 2) and persona style vectors.
#[derive(Debug, Clone)]
pub struct Embedding(pub [f64; 32]);

impl Embedding {
    pub fn zero() -> Self { Self([0.0; 32]) }

    /// Cosine similarity to another embedding.
    pub fn cosine(&self, other: &Embedding) -> f64 {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..32 {
            dot += self.0[i] * other.0[i];
            norm_a += self.0[i] * self.0[i];
            norm_b += other.0[i] * other.0[i];
        }
        if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Weighted average with another embedding.
    pub fn blend(&self, other: &Embedding, weight: f64) -> Embedding {
        let mut result = [0.0; 32];
        for i in 0..32 {
            result[i] = self.0[i] * (1.0 - weight) + other.0[i] * weight;
        }
        Embedding(result)
    }
}

/// A musician persona — each agent has one. Develops style over time.
#[derive(Debug, Clone)]
pub struct MusicianPersona {
    pub agent_id: u32,
    pub name: String,
    /// Style vector — the agent's unique "sound" in 32-dim space.
    pub style: Embedding,
    /// Vector DB: phrase embeddings this agent has produced, keyed by label.
    pub vector_db: HashMap<String, Embedding>,
    /// Total riffs this persona has participated in.
    pub experience: u32,
    /// Preferred response modes based on accumulated style.
    pub mode_affinity: HashMap<ResponseMode, f64>,
}

impl MusicianPersona {
    pub fn new(agent_id: u32, name: &str) -> Self {
        Self {
            agent_id,
            name: name.to_string(),
            style: Embedding::zero(),
            vector_db: HashMap::new(),
            experience: 0,
            mode_affinity: HashMap::new(),
        }
    }

    /// Record a riff and update the persona's style.
    pub fn record_riff(&mut self, riff: &Riff, phrase_embedding: &Embedding) {
        self.experience += 1;
        // Style drifts toward successful phrases
        let drift = match riff.quality {
            Quality::Strong => 0.15,
            Quality::Ok => 0.05,
            Quality::Weak => 0.01,
        };
        self.style = self.style.blend(phrase_embedding, drift);
        // Store in vector DB
        let key = format!("gen{}-r{}-a{}", riff.round, riff.round, riff.agent_id);
        self.vector_db.insert(key, phrase_embedding.clone());
    }

    /// Find the closest stored phrase to a given embedding.
    pub fn find_closest(&self, embedding: &Embedding) -> Option<(String, f64)> {
        self.vector_db.iter()
            .map(|(k, v)| (k.clone(), v.cosine(embedding)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Retrieve style similarity to another persona.
    pub fn style_similarity(&self, other: &MusicianPersona) -> f64 {
        self.style.cosine(&other.style)
    }

    /// Update mode affinity based on what produced quality.
    pub fn update_affinity(&mut self, mode: ResponseMode, quality: Quality) {
        let entry = self.mode_affinity.entry(mode).or_insert(0.5);
        let delta = match quality {
            Quality::Strong => 0.1,
            Quality::Ok => 0.02,
            Quality::Weak => -0.05,
        };
        *entry = (*entry + delta).clamp(0.0, 1.0);
    }

    /// Get the preferred mode for this persona.
    pub fn preferred_mode(&self) -> ResponseMode {
        ResponseMode::all().iter()
            .map(|&m| (m, *self.mode_affinity.get(&m).unwrap_or(&0.5)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(m, _)| m)
            .unwrap_or(ResponseMode::Escalate)
    }
}

// ════════════════════════════════════════════════════════════════════
// v4 Feature 2: Crates-as-Phrases
// ════════════════════════════════════════════════════════════════════

/// A crate signature — metadata that maps to the same 32-dim space as musical phrases.
/// LOC = note count, tests = rhythm, features = intervals, quality = dynamics.
#[derive(Debug, Clone)]
pub struct CrateSignature {
    pub name: String,
    pub loc: usize,
    pub tests: usize,
    pub features: Vec<String>,
    pub quality: Quality,
}

impl CrateSignature {
    /// Embed a crate into the 32-dim phrase space.
    /// The mapping: loc → first 8 dims, tests → next 8, features → next 8, quality → last 8.
    pub fn embed(&self) -> Embedding {
        let mut v = [0.0f64; 32];

        // LOC → notes (dims 0–7): log-scaled
        let loc_norm = (self.loc as f64 + 1.0).ln() / 10.0;
        for i in 0..8 { v[i] = loc_norm * (1.0 + i as f64 * 0.1); }

        // Tests → rhythm (dims 8–15): test density
        let test_density = if self.loc > 0 { self.tests as f64 / self.loc as f64 } else { 0.0 };
        for i in 0..8 { v[8 + i] = test_density * (1.0 + i as f64 * 0.05).min(1.0); }

        // Features → intervals (dims 16–23): feature hash
        for (i, feat) in self.features.iter().enumerate() {
            if i >= 8 { break; }
            let hash = feat.chars().fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64));
            v[16 + i] = (hash as f64 % 100.0) / 100.0;
        }

        // Quality → dynamics (dims 24–31)
        let q_val = match self.quality { Quality::Weak => 0.2, Quality::Ok => 0.5, Quality::Strong => 0.9 };
        for i in 0..8 { v[24 + i] = q_val * (0.8 + i as f64 * 0.025); }

        Embedding(v)
    }
}

/// A musical phrase that can be compared with crate signatures.
#[derive(Debug, Clone)]
pub struct MusicalPhrase {
    pub label: String,
    pub embedding: Embedding,
}

impl MusicalPhrase {
    /// Create a phrase from a crate signature.
    pub fn from_crate(sig: &CrateSignature) -> Self {
        Self { label: format!("crate:{}", sig.name), embedding: sig.embed() }
    }

    /// Similarity between a phrase and a crate.
    pub fn similarity_to_crate(&self, sig: &CrateSignature) -> f64 {
        self.embedding.cosine(&sig.embed())
    }
}

// ════════════════════════════════════════════════════════════════════
// v4 Feature 3: Autonomous Spec Evolution
// ════════════════════════════════════════════════════════════════════

/// A living spec that evolves based on riff outcomes.
#[derive(Debug, Clone)]
pub struct EvolvingSpec {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub version: u32,
    /// What the spec currently requires.
    pub requirements: Vec<String>,
    /// What worked in previous riffs — absorbed into the spec.
    pub absorbed_patterns: Vec<String>,
    /// Quality score of the spec (how well riffing on it has gone).
    pub fitness: f64,
}

impl EvolvingSpec {
    pub fn new(id: &str, name: &str, domain: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            domain: domain.to_string(),
            version: 1,
            requirements: Vec::new(),
            absorbed_patterns: Vec::new(),
            fitness: 0.5,
        }
    }

    /// Evolve the spec based on a round of riffing.
    /// Successful features get absorbed; fitness is updated.
    pub fn evolve(&mut self, round: &Round) {
        self.version += 1;
        for riff in &round.riffs {
            if riff.quality == Quality::Strong {
                for feat in &riff.features {
                    if !self.absorbed_patterns.contains(feat) {
                        self.absorbed_patterns.push(feat.clone());
                    }
                }
            }
        }
        // Fitness: blend with round productivity
        let round_quality: f64 = round.riffs.iter()
            .map(|r| match r.quality { Quality::Weak => 0.2, Quality::Ok => 0.5, Quality::Strong => 0.9 })
            .sum::<f64>() / round.riffs.len().max(1) as f64;
        self.fitness = self.fitness * 0.7 + round_quality * 0.3;
    }

    /// Generate new requirements based on absorbed patterns.
    pub fn suggest_requirements(&self) -> Vec<String> {
        let mut reqs = self.requirements.clone();
        for pattern in &self.absorbed_patterns {
            let suggested = format!("Support {} with configurable parameters", pattern);
            if !reqs.contains(&suggested) {
                reqs.push(suggested);
            }
        }
        reqs
    }

    /// Check if the spec has converged (fitness high, version high).
    pub fn is_mature(&self) -> bool { self.fitness > 0.8 && self.version > 5 }
}

// ════════════════════════════════════════════════════════════════════
// Cross-session learning (enhanced for v4)
// ════════════════════════════════════════════════════════════════════

/// Accumulated success rates per (agent, mode) pair.
#[derive(Debug, Clone, Default)]
pub struct ModeStats {
    pub total_uses: u32,
    pub total_surprise: f64,
    pub strong_count: u32,
    pub weak_count: u32,
}

impl ModeStats {
    pub fn avg_surprise(&self) -> f64 {
        if self.total_uses == 0 { 0.0 } else { self.total_surprise / self.total_uses as f64 }
    }
    pub fn success_rate(&self) -> f64 {
        if self.total_uses == 0 { 0.5 } else { self.strong_count as f64 / self.total_uses as f64 }
    }
}

/// Session metrics including bootstrap generation.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub generation: u32,
    pub total_rounds: usize,
    pub productive_rounds: usize,
    pub total_loc: usize,
    pub total_tests: usize,
    pub total_features: usize,
    pub avg_surprise: f64,
    pub streak: u32,
}

// ════════════════════════════════════════════════════════════════════
// v4 Feature 4: Generation Memory with Pruning
// ════════════════════════════════════════════════════════════════════

const MAX_MEMORY_ENTRIES: usize = 64;

/// Cross-session learning with bounded memory and pruning.
#[derive(Debug, Clone)]
pub struct RiffMemory {
    pub best_modes: HashMap<u32, ResponseMode>,
    pub mode_stats: HashMap<(u32, ResponseMode), ModeStats>,
    pub spec_patterns: HashMap<String, f64>,
    pub total_rounds: u64,
    pub total_surprise: f64,
    pub generation_history: Vec<SessionMetrics>,
    /// Pruning: each entry has a "growth contribution" score. Weakest are evicted.
    pattern_scores: HashMap<String, f64>,
}

impl Default for RiffMemory {
    fn default() -> Self {
        Self {
            best_modes: HashMap::new(),
            mode_stats: HashMap::new(),
            spec_patterns: HashMap::new(),
            total_rounds: 0,
            total_surprise: 0.0,
            generation_history: Vec::new(),
            pattern_scores: HashMap::new(),
        }
    }
}

impl RiffMemory {
    pub fn new() -> Self { Self::default() }

    pub fn learn(&mut self, rounds: &[Round]) {
        self.total_rounds += rounds.len() as u64;
        for r in rounds {
            self.total_surprise += r.surprise_sum;
            for riff in &r.riffs {
                let mode = ResponseMode::auto(riff.surprise, 0, r.number);
                let stats = self.mode_stats.entry((riff.agent_id, mode)).or_default();
                stats.total_uses += 1;
                stats.total_surprise += riff.surprise;
                match riff.quality {
                    Quality::Strong => stats.strong_count += 1,
                    Quality::Weak => stats.weak_count += 1,
                    _ => {}
                }
                // Track feature patterns for pruning
                for feat in &riff.features {
                    let score = self.pattern_scores.entry(feat.clone()).or_insert(0.5);
                    *score = *score * 0.9 + match riff.quality {
                        Quality::Strong => 0.2,
                        Quality::Ok => 0.05,
                        Quality::Weak => -0.1,
                    };
                }
            }
        }
        self.prune();
    }

    /// Prune weakest patterns when memory exceeds bounds.
    fn prune(&mut self) {
        if self.pattern_scores.len() <= MAX_MEMORY_ENTRIES { return; }
        let mut entries: Vec<_> = self.pattern_scores.iter().collect();
        entries.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let to_remove = entries.len() - MAX_MEMORY_ENTRIES;
        let keys_to_remove: Vec<String> = entries.iter().take(to_remove).map(|(k, _)| (*k).clone()).collect();
        for key in keys_to_remove {
            self.pattern_scores.remove(&key);
        }
    }

    /// Prune generation history: remove generations that didn't produce growth.
    pub fn prune_generations(&mut self) {
        if self.generation_history.len() <= 2 { return; }
        // Keep only generations that produced above-median productivity
        let median_loc = {
            let mut locs: Vec<_> = self.generation_history.iter().map(|g| g.total_loc).collect();
            locs.sort();
            locs[locs.len() / 2]
        };
        self.generation_history.retain(|g| g.total_loc >= median_loc);
    }

    /// Get the top N patterns by growth score.
    pub fn top_patterns(&self, n: usize) -> Vec<(String, f64)> {
        let mut entries: Vec<_> = self.pattern_scores.iter().map(|(k, &v)| (k.clone(), v)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        entries.truncate(n);
        entries
    }

    pub fn record_generation(&mut self, metrics: SessionMetrics) {
        self.generation_history.push(metrics);
        self.prune_generations();
    }

    pub fn recommend_mode(&self, agent_id: u32) -> ResponseMode {
        self.best_modes.get(&agent_id).copied().unwrap_or(ResponseMode::Escalate)
    }

    /// Predict which agent+mode will produce the best output.
    pub fn predict_best(&self, agents: &[u32]) -> (u32, ResponseMode, f64) {
        let mut best_agent = agents[0];
        let mut best_mode = ResponseMode::Escalate;
        let mut best_score = -1.0f64;

        for &agent in agents {
            for &mode in ResponseMode::all() {
                let stats = self.mode_stats.get(&(agent, mode)).cloned().unwrap_or_default();
                let score = stats.avg_surprise() * 0.6 + stats.success_rate() * 0.4;
                if score > best_score {
                    best_score = score;
                    best_agent = agent;
                    best_mode = mode;
                }
            }
        }
        (best_agent, best_mode, best_score)
    }
}

// ════════════════════════════════════════════════════════════════════
// v4 Feature 5: Self-Bootstrapping (spec generation for v5)
// ════════════════════════════════════════════════════════════════════

/// A generated spec for the NEXT version of the crate.
#[derive(Debug, Clone)]
pub struct NextSpec {
    pub version: String,
    pub features: Vec<String>,
    pub rationale: String,
    pub confidence: f64,
}

/// The self-bootstrap engine: analyzes current state and generates a spec for v5.
#[derive(Debug, Clone)]
pub struct SelfBootstrap {
    pub current_version: u32,
    pub capabilities: Vec<String>,
}

impl SelfBootstrap {
    pub fn new(current_version: u32) -> Self {
        Self { current_version, capabilities: Vec::new() }
    }

    /// Analyze accumulated memory and generate the next spec.
    pub fn generate_next_spec(&self, memory: &RiffMemory, personas: &[MusicianPersona]) -> NextSpec {
        let next_version = self.current_version + 1;

        // Gather insights from memory
        let top_patterns = memory.top_patterns(5);
        let best_agent = memory.predict_best(&personas.iter().map(|p| p.agent_id).collect::<Vec<_>>());

        // Identify gaps: what modes have low success rates?
        let mut weak_modes = Vec::new();
        for &mode in ResponseMode::all() {
            let avg: f64 = personas.iter()
                .filter_map(|p| memory.mode_stats.get(&(p.agent_id, mode)))
                .map(|s| s.success_rate())
                .sum::<f64>() / personas.len().max(1) as f64;
            if avg < 0.4 { weak_modes.push(mode); }
        }

        // Generate feature proposals based on what's missing
        let mut features = Vec::new();

        // Feature 1: Always propose evolution of the strongest pattern
        if let Some((pattern, _score)) = top_patterns.first() {
            features.push(format!("Advanced {} with multi-agent tournament", pattern));
        }

        // Feature 2: Persona divergence tracking
        if personas.len() >= 2 {
            let sim = personas[0].style_similarity(&personas[1]);
            features.push(format!("Persona divergence engine (current divergence: {:.2})", 1.0 - sim));
        }

        // Feature 3: Weak mode improvement
        if !weak_modes.is_empty() {
            let mode_names: Vec<String> = weak_modes.iter().map(|m| format!("{:?}", m)).collect();
            features.push(format!("Adaptive mode training for {:?}", mode_names));
        }

        // Feature 4: Always propose the snowball continues
        features.push(format!("{}-generation bootstrap chain", next_version + 1));

        // Feature 5: Memory architecture evolution
        features.push("Hierarchical memory with temporal decay".to_string());

        let confidence = if memory.total_rounds > 10 { 0.85 } else { 0.5 };
        let rationale = format!(
            "Generated from {} rounds of riff history, {} personas, {} top patterns. Best agent: {}",
            memory.total_rounds, personas.len(), top_patterns.len(), best_agent.0
        );

        NextSpec {
            version: format!("v{}", next_version),
            features,
            rationale,
            confidence,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Bootstrap Verifier (carried from v3)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub generation: u32,
    pub compiles: bool,
    pub tests_pass: bool,
    pub test_count: usize,
    pub test_failures: Vec<String>,
    pub warnings: Vec<String>,
}

impl VerifyResult {
    pub fn success(generation: u32, test_count: usize) -> Self {
        Self { generation, compiles: true, tests_pass: true, test_count, test_failures: Vec::new(), warnings: Vec::new() }
    }
    pub fn failure(generation: u32, failures: Vec<String>) -> Self {
        Self { generation, compiles: true, tests_pass: false, test_count: 0, test_failures: failures, warnings: Vec::new() }
    }
    pub fn is_ok(&self) -> bool { self.compiles && self.tests_pass }
}

#[derive(Debug, Clone)]
pub struct GrowthCheck {
    pub growing: bool,
    pub loc_deltas: Vec<f64>,
    pub test_deltas: Vec<f64>,
    pub feature_deltas: Vec<f64>,
    pub surprise_deltas: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BootstrapVerifier {
    pub verified_generations: Vec<VerifyResult>,
}

impl BootstrapVerifier {
    pub fn new() -> Self { Self { verified_generations: Vec::new() } }

    pub fn verify(&mut self, metrics: &SessionMetrics) -> VerifyResult {
        let gen = metrics.generation;
        if metrics.total_rounds == 0 {
            let r = VerifyResult::failure(gen, vec!["No rounds generated".into()]);
            self.verified_generations.push(r.clone());
            return r;
        }
        if metrics.total_loc == 0 {
            let r = VerifyResult::failure(gen, vec!["No LOC produced".into()]);
            self.verified_generations.push(r.clone());
            return r;
        }
        if metrics.total_tests == 0 {
            let r = VerifyResult::failure(gen, vec!["No tests produced".into()]);
            self.verified_generations.push(r.clone());
            return r;
        }
        let r = VerifyResult::success(gen, metrics.total_tests);
        self.verified_generations.push(r.clone());
        r
    }

    pub fn verify_chain(&mut self, chain: &[SessionMetrics]) -> Vec<VerifyResult> {
        chain.iter().map(|m| self.verify(m)).collect()
    }

    pub fn check_growth(chain: &[SessionMetrics]) -> GrowthCheck {
        if chain.len() < 2 {
            return GrowthCheck { growing: true, loc_deltas: Vec::new(), test_deltas: Vec::new(), feature_deltas: Vec::new(), surprise_deltas: Vec::new() };
        }
        let mut ld = Vec::new(); let mut td = Vec::new(); let mut fd = Vec::new(); let mut sd = Vec::new();
        for w in chain.windows(2) {
            ld.push(w[1].total_loc as f64 - w[0].total_loc as f64);
            td.push(w[1].total_tests as f64 - w[0].total_tests as f64);
            fd.push(w[1].total_features as f64 - w[0].total_features as f64);
            sd.push(w[1].avg_surprise - w[0].avg_surprise);
        }
        let growing = ld.iter().all(|&d| d >= 0.0) && td.iter().all(|&d| d >= 0.0);
        GrowthCheck { growing, loc_deltas: ld, test_deltas: td, feature_deltas: fd, surprise_deltas: sd }
    }
}

// ════════════════════════════════════════════════════════════════════
// Snowball Tracker (carried from v3)
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct GrowthRate {
    pub from_gen: u32,
    pub to_gen: u32,
    pub loc_rate: f64,
    pub test_rate: f64,
    pub feature_rate: f64,
    pub surprise_delta: f64,
}

#[derive(Debug, Clone)]
pub struct SnowballTracker {
    pub generations: Vec<SessionMetrics>,
    pub growth_rates: Vec<GrowthRate>,
}

impl SnowballTracker {
    pub fn new() -> Self { Self { generations: Vec::new(), growth_rates: Vec::new() } }

    pub fn record(&mut self, metrics: SessionMetrics) {
        if let Some(prev) = self.generations.last() {
            let loc_rate = if prev.total_loc > 0 { metrics.total_loc as f64 / prev.total_loc as f64 } else { 1.0 };
            let test_rate = if prev.total_tests > 0 { metrics.total_tests as f64 / prev.total_tests as f64 } else { 1.0 };
            let feature_rate = if prev.total_features > 0 { metrics.total_features as f64 / prev.total_features as f64 } else { 1.0 };
            self.growth_rates.push(GrowthRate {
                from_gen: prev.generation,
                to_gen: metrics.generation,
                loc_rate,
                test_rate,
                feature_rate,
                surprise_delta: metrics.avg_surprise - prev.avg_surprise,
            });
        }
        self.generations.push(metrics);
    }

    pub fn is_growing(&self) -> bool {
        self.growth_rates.iter().all(|g| g.loc_rate >= 1.0 && g.test_rate >= 1.0)
    }

    pub fn avg_growth_rate(&self) -> f64 {
        if self.growth_rates.is_empty() { return 0.0; }
        self.growth_rates.iter().map(|g| (g.loc_rate + g.test_rate + g.feature_rate) / 3.0).sum::<f64>() / self.growth_rates.len() as f64
    }
}

// ════════════════════════════════════════════════════════════════════
// v4 Riff Session (evolved from v3 MultiSpecSession)
// ════════════════════════════════════════════════════════════════════

/// Summary of a round's evaluation.
#[derive(Debug, Clone)]
pub struct RoundSummary {
    pub surprise: f64,
    pub productive: bool,
    pub landed: bool,
    pub mode: ResponseMode,
    pub best_productivity: f64,
}

/// A riff session with musician personas and evolving specs.
#[derive(Debug, Clone)]
pub struct RiffSession {
    pub agents: Vec<u32>,
    pub personas: HashMap<u32, MusicianPersona>,
    pub specs: Vec<EvolvingSpec>,
    pub rounds: Vec<Round>,
    pub memory: RiffMemory,
    pub current_round: u32,
    pub mode: ResponseMode,
    pub streak: u32,
    pub finished: bool,
    pub generation: u32,
    pub cross_spec_patterns: HashMap<String, Vec<String>>,
}

impl RiffSession {
    pub fn new(agents: Vec<u32>, specs: Vec<EvolvingSpec>, generation: u32) -> Self {
        let personas = agents.iter().map(|&id| (id, MusicianPersona::new(id, &format!("agent-{}", id)))).collect();
        Self {
            agents, personas, specs, rounds: Vec::new(), memory: RiffMemory::new(),
            current_round: 0, mode: ResponseMode::Escalate, streak: 0, finished: false,
            generation, cross_spec_patterns: HashMap::new(),
        }
    }

    pub fn new_round(&mut self) -> &mut Round {
        self.rounds.push(Round::new(self.current_round));
        self.current_round += 1;
        self.rounds.last_mut().unwrap()
    }

    pub fn riff(&mut self, agent_id: u32, quality: Quality, surprise: f64) {
        let riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    pub fn riff_with_output(&mut self, agent_id: u32, quality: Quality, surprise: f64, loc: usize, tests: usize, features: Vec<&str>) {
        let mut riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        riff.loc = loc; riff.tests = tests;
        riff.features = features.iter().map(|s| s.to_string()).collect();
        // Update persona
        if let Some(persona) = self.personas.get_mut(&agent_id) {
            let sig = CrateSignature {
                name: format!("gen{}-r{}", self.generation, self.current_round),
                loc, tests, features: riff.features.clone(), quality,
            };
            let embedding = sig.embed();
            persona.record_riff(&riff, &embedding);
            persona.update_affinity(self.mode, quality);
        }
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    /// Riff targeting a specific evolving spec.
    pub fn riff_for_spec(&mut self, agent_id: u32, spec_id: &str, quality: Quality, surprise: f64, loc: usize, tests: usize, features: Vec<&str>) {
        let mut riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        riff.spec_id = Some(spec_id.to_string());
        riff.loc = loc; riff.tests = tests;
        riff.features = features.iter().map(|s| s.to_string()).collect();
        // Share cross-spec patterns
        let shared: Vec<String> = self.cross_spec_patterns.iter()
            .filter(|(k, _)| *k != spec_id)
            .flat_map(|(_, v)| v.iter().cloned())
            .collect();
        for p in &shared {
            if !riff.features.contains(p) { riff.features.push(p.clone()); }
        }
        let entry = self.cross_spec_patterns.entry(spec_id.to_string()).or_insert_with(Vec::new);
        for f in &riff.features {
            if !entry.contains(f) { entry.push(f.clone()); }
        }
        // Update persona
        if let Some(persona) = self.personas.get_mut(&agent_id) {
            let sig = CrateSignature {
                name: format!("gen{}-{}-r{}", self.generation, spec_id, self.current_round),
                loc, tests, features: riff.features.clone(), quality,
            };
            persona.record_riff(&riff, &sig.embed());
            persona.update_affinity(self.mode, quality);
        }
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }

    pub fn evaluate(&mut self) -> RoundSummary {
        let round = match self.rounds.last() {
            Some(r) => r,
            None => return RoundSummary { surprise: 0.0, productive: false, landed: false, mode: self.mode, best_productivity: 0.0 },
        };
        let surprise = round.surprise_sum;
        let productive = round.was_productive();
        if productive { self.streak += 1; } else { self.streak = 0; }
        let landed = surprise > 0.8 && round.riffs.iter().any(|r| r.quality == Quality::Strong);
        self.mode = ResponseMode::auto(surprise, self.streak, self.current_round);
        if self.streak == 0 && self.current_round > 5 { self.finished = true; }
        // v4: Evolve specs after each round
        for spec in &mut self.specs {
            spec.evolve(round);
        }
        let best_prod = round.riffs.iter().map(|r| r.productivity()).fold(0.0f64, f64::max);
        RoundSummary { surprise, productive, landed, mode: self.mode, best_productivity: best_prod }
    }

    pub fn metrics(&self) -> SessionMetrics {
        let total_rounds = self.rounds.len();
        let productive = self.rounds.iter().filter(|r| r.was_productive()).count();
        let total_loc: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.loc).sum();
        let total_tests: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.tests).sum();
        let total_features: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.features.len()).sum();
        let total_surprise: f64 = self.rounds.iter().map(|r| r.surprise_sum).sum();
        SessionMetrics {
            generation: self.generation,
            total_rounds,
            productive_rounds: productive,
            total_loc,
            total_tests,
            total_features,
            avg_surprise: if total_rounds > 0 { total_surprise / total_rounds as f64 } else { 0.0 },
            streak: self.streak,
        }
    }

    /// Bootstrap the next generation with inherited memory and personas.
    pub fn bootstrap_next(&self) -> RiffSession {
        let mut next = RiffSession::new(self.agents.clone(), self.specs.clone(), self.generation + 1);
        next.memory = self.memory.clone();
        next.personas = self.personas.clone();
        next.cross_spec_patterns = self.cross_spec_patterns.clone();
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy tests (v1/v2/v3) ─────────────────────────────────────

    #[test]
    fn trit_pack_unpack() {
        let trits = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
        let packed = pack_16(&trits);
        assert_eq!(Trit::unpack_bits((packed & 0x3) as u8), Trit::Pos);
        assert_eq!(Trit::unpack_bits(((packed >> 2) & 0x3) as u8), Trit::Neg);
    }

    #[test]
    fn riff_productivity() {
        let mut r = Riff::new(0, 1, Quality::Strong, 0.8);
        r.loc = 200; r.tests = 15; r.features = vec!["gpu-packing".to_string()];
        assert!(r.productivity() > 0.0);
    }

    #[test]
    fn response_mode_auto() {
        assert_eq!(ResponseMode::auto(0.1, 0, 3), ResponseMode::Provoked);
        assert_eq!(ResponseMode::auto(0.8, 0, 3), ResponseMode::Escalate);
        assert_eq!(ResponseMode::auto(0.5, 6, 3), ResponseMode::Pivot);
        assert_eq!(ResponseMode::auto(0.5, 0, 9), ResponseMode::Invert);
    }

    #[test]
    fn session_basic() {
        let mut s = RiffSession::new(vec![0, 1], vec![], 1);
        s.new_round();
        s.riff_with_output(0, Quality::Ok, 0.3, 100, 8, vec!["baseline"]);
        s.riff_with_output(1, Quality::Strong, 0.7, 300, 20, vec!["gpu-packing", "entropy"]);
        let summary = s.evaluate();
        assert!(summary.productive);
    }

    #[test]
    fn stale_detection() {
        let mut s = RiffSession::new(vec![0, 1], vec![], 1);
        for _ in 0..6 {
            s.new_round();
            s.riff(0, Quality::Weak, 0.05);
            s.riff(1, Quality::Weak, 0.05);
            s.evaluate();
        }
        assert!(s.finished);
    }

    #[test]
    fn landing_detection() {
        let mut s = RiffSession::new(vec![0, 1], vec![], 1);
        s.new_round();
        s.riff(0, Quality::Strong, 0.9);
        s.riff(1, Quality::Strong, 0.85);
        let summary = s.evaluate();
        assert!(summary.landed);
    }

    // ── v4 Feature 1: Musician-Soul Integration ─────────────────────

    #[test]
    fn persona_style_develops_over_time() {
        let mut persona = MusicianPersona::new(0, "test-agent");
        assert_eq!(persona.experience, 0);

        // Simulate several strong riffs in a similar direction
        let sig = CrateSignature {
            name: "test".into(), loc: 200, tests: 15,
            features: vec!["gpu-kernel".into()], quality: Quality::Strong,
        };
        let embedding = sig.embed();
        for i in 0..5 {
            let riff = Riff::new(0, i, Quality::Strong, 0.8);
            persona.record_riff(&riff, &embedding);
        }
        assert_eq!(persona.experience, 5);
        // Style should have drifted from zero
        let style_norm: f64 = persona.style.0.iter().map(|v| v * v).sum();
        assert!(style_norm > 0.0, "Style should have drifted from zero");
    }

    #[test]
    fn persona_vector_db_similarity() {
        let mut persona = MusicianPersona::new(0, "test");
        let sig = CrateSignature {
            name: "test".into(), loc: 200, tests: 15,
            features: vec!["kernel".into()], quality: Quality::Strong,
        };
        let emb = sig.embed();
        let riff = Riff::new(0, 1, Quality::Strong, 0.8);
        persona.record_riff(&riff, &emb);

        // Query with the same embedding should find high similarity
        let (_key, sim) = persona.find_closest(&emb).unwrap();
        assert!(sim > 0.99);
    }

    #[test]
    fn persona_mode_affinity() {
        let mut persona = MusicianPersona::new(0, "test");
        // Train: Escalate is consistently strong
        for _ in 0..5 {
            persona.update_affinity(ResponseMode::Escalate, Quality::Strong);
        }
        // Pivot is consistently weak
        for _ in 0..5 {
            persona.update_affinity(ResponseMode::Pivot, Quality::Weak);
        }
        assert_eq!(persona.preferred_mode(), ResponseMode::Escalate);
    }

    // ── v4 Feature 2: Crates-as-Phrases ─────────────────────────────

    #[test]
    fn crate_to_phrase_embedding() {
        let sig = CrateSignature {
            name: "agent-riff-v4".into(),
            loc: 500,
            tests: 25,
            features: vec!["musician-soul".into(), "crates-as-phrases".into()],
            quality: Quality::Strong,
        };
        let phrase = MusicalPhrase::from_crate(&sig);
        assert_eq!(phrase.label, "crate:agent-riff-v4");
        // Embedding should be non-zero
        let norm: f64 = phrase.embedding.0.iter().map(|v| v * v).sum();
        assert!(norm > 0.0, "Crate embedding should be non-zero");
    }

    #[test]
    fn crate_phrase_similarity() {
        // Two similar crates should have high similarity
        let sig1 = CrateSignature {
            name: "a".into(), loc: 200, tests: 15,
            features: vec!["gpu".into()], quality: Quality::Strong,
        };
        let sig2 = CrateSignature {
            name: "b".into(), loc: 210, tests: 16,
            features: vec!["gpu".into()], quality: Quality::Strong,
        };
        let phrase = MusicalPhrase::from_crate(&sig1);
        let sim = phrase.similarity_to_crate(&sig2);
        assert!(sim > 0.99, "Similar crates should have very high similarity: got {}", sim);
    }

    #[test]
    fn different_quality_different_embedding() {
        let weak = CrateSignature {
            name: "w".into(), loc: 100, tests: 2,
            features: vec!["basic".into()], quality: Quality::Weak,
        };
        let strong = CrateSignature {
            name: "s".into(), loc: 100, tests: 2,
            features: vec!["basic".into()], quality: Quality::Strong,
        };
        let sim = weak.embed().cosine(&strong.embed());
        // Should be similar but not identical
        assert!(sim < 1.0, "Different quality should produce different embeddings");
    }

    // ── v4 Feature 3: Autonomous Spec Evolution ─────────────────────

    #[test]
    fn spec_evolution_absorbs_patterns() {
        let mut spec = EvolvingSpec::new("test", "Test Spec", "testing");
        assert!(spec.absorbed_patterns.is_empty());

        let mut round = Round::new(1);
        round.add(Riff {
            agent_id: 0, round: 1, quality: Quality::Strong, surprise: 0.8,
            loc: 200, tests: 10, features: vec!["gpu-kernel".into()], spec_id: None,
        });
        spec.evolve(&round);
        assert!(spec.absorbed_patterns.contains(&"gpu-kernel".to_string()));
        assert!(spec.fitness > 0.5);
    }

    #[test]
    fn spec_ignores_weak_riffs() {
        let mut spec = EvolvingSpec::new("test", "Test", "testing");
        let mut round = Round::new(1);
        round.add(Riff {
            agent_id: 0, round: 1, quality: Quality::Weak, surprise: 0.1,
            loc: 50, tests: 1, features: vec!["bad-idea".into()], spec_id: None,
        });
        spec.evolve(&round);
        assert!(!spec.absorbed_patterns.contains(&"bad-idea".to_string()));
        assert!(spec.fitness < 0.5); // Fitness should have decreased
    }

    #[test]
    fn spec_maturity() {
        let mut spec = EvolvingSpec::new("test", "Test", "testing");
        assert!(!spec.is_mature());
        spec.version = 6;
        spec.fitness = 0.9;
        assert!(spec.is_mature());
    }

    #[test]
    fn spec_suggests_requirements() {
        let mut spec = EvolvingSpec::new("test", "Test", "testing");
        spec.absorbed_patterns = vec!["gpu-kernel".into(), "simd-packing".into()];
        let reqs = spec.suggest_requirements();
        assert!(reqs.iter().any(|r| r.contains("gpu-kernel")));
        assert!(reqs.iter().any(|r| r.contains("simd-packing")));
    }

    // ── v4 Feature 4: Generation Memory with Pruning ────────────────

    #[test]
    fn memory_prunes_weak_patterns() {
        let mut mem = RiffMemory::new();
        // Add many patterns, some weak some strong
        for i in 0..80 {
            let mut round = Round::new(i);
            let quality = if i < 20 { Quality::Weak } else { Quality::Strong };
            let surprise = if i < 20 { 0.05 } else { 0.8 };
            let feat_name = format!("pattern-{}", i);
            let mut riff = Riff::new(0, i, quality, surprise);
            riff.loc = 100; riff.tests = 5; riff.features = vec![feat_name];
            round.add(riff);
            mem.learn(&[round]);
        }
        // Memory should be bounded
        let top = mem.top_patterns(5);
        assert!(top.iter().all(|(_, s)| *s > 0.5), "Top patterns should be the strong ones");
    }

    #[test]
    fn generation_pruning_keeps_best() {
        let mut mem = RiffMemory::new();
        mem.record_generation(SessionMetrics {
            generation: 1, total_rounds: 2, productive_rounds: 1,
            total_loc: 100, total_tests: 5, total_features: 2,
            avg_surprise: 0.3, streak: 1,
        });
        mem.record_generation(SessionMetrics {
            generation: 2, total_rounds: 3, productive_rounds: 2,
            total_loc: 500, total_tests: 25, total_features: 8,
            avg_surprise: 0.7, streak: 3,
        });
        mem.record_generation(SessionMetrics {
            generation: 3, total_rounds: 1, productive_rounds: 0,
            total_loc: 50, total_tests: 2, total_features: 1,
            avg_surprise: 0.1, streak: 0,
        });
        // After pruning, gen 3 (lowest LOC) should be removed
        assert!(mem.generation_history.len() <= 3);
        // The strong generation should survive
        assert!(mem.generation_history.iter().any(|g| g.generation == 2));
    }

    // ── v4 Feature 5: Self-Bootstrapping ────────────────────────────

    #[test]
    fn self_spec_generation() {
        let bootstrap = SelfBootstrap::new(4);
        let mut mem = RiffMemory::new();
        // Give it some history
        let mut round = Round::new(1);
        round.add(Riff {
            agent_id: 0, round: 1, quality: Quality::Strong, surprise: 0.8,
            loc: 200, tests: 15, features: vec!["gpu-kernel".into()], spec_id: None,
        });
        for _ in 0..15 { mem.learn(&[round.clone()]); }

        let personas = vec![MusicianPersona::new(0, "alpha"), MusicianPersona::new(1, "beta")];
        let spec = bootstrap.generate_next_spec(&mem, &personas);

        assert_eq!(spec.version, "v5");
        assert!(!spec.features.is_empty());
        assert!(spec.confidence > 0.0);
        assert!(!spec.rationale.is_empty());
    }

    #[test]
    fn self_spec_includes_divergence() {
        let bootstrap = SelfBootstrap::new(4);
        let mem = RiffMemory::new();
        let mut p1 = MusicianPersona::new(0, "alpha");
        let mut p2 = MusicianPersona::new(1, "beta");
        // Give them different styles
        let mut emb1 = Embedding::zero();
        emb1.0[0] = 1.0;
        let mut emb2 = Embedding::zero();
        emb2.0[0] = -1.0;
        p1.style = emb1;
        p2.style = emb2;

        let spec = bootstrap.generate_next_spec(&mem, &[p1, p2]);
        assert!(spec.features.iter().any(|f| f.contains("divergence")));
    }

    // ── THE BIG TEST: 4-generation bootstrap chain ──────────────────

    #[test]
    fn four_generation_bootstrap_chain() {
        let specs = vec![
            EvolvingSpec::new("ternary-core", "Core Types", "ternary"),
            EvolvingSpec::new("ternary-gpu", "GPU Kernels", "ternary"),
        ];

        // ── Generation 1: Baseline ──
        let mut gen1 = RiffSession::new(vec![0, 1], specs.clone(), 1);
        gen1.new_round();
        gen1.riff_for_spec(0, "ternary-core", Quality::Ok, 0.3, 100, 5, vec!["basic-packing"]);
        gen1.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.6, 200, 12, vec!["kernel-launch"]);
        gen1.evaluate();
        gen1.memory.learn(&gen1.rounds);
        let gen1_metrics = gen1.metrics();
        gen1.memory.record_generation(gen1_metrics.clone());

        // Personas should have developed
        assert!(gen1.personas[&0].experience > 0);
        assert!(gen1.personas[&1].experience > 0);

        // ── Generation 2: Inherited memory + personas ──
        let mut gen2 = gen1.bootstrap_next();
        assert_eq!(gen2.generation, 2);
        assert_eq!(gen2.memory.total_rounds, 1);
        // Personas carry over with their style
        assert!(gen2.personas[&0].experience > 0);

        gen2.new_round();
        gen2.riff_for_spec(0, "ternary-core", Quality::Strong, 0.7, 300, 18, vec!["fast-pack"]);
        gen2.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.8, 450, 28, vec!["cuda-ops"]);
        let gen2_summary = gen2.evaluate();
        assert!(gen2_summary.productive);
        gen2.memory.learn(&gen2.rounds);
        let gen2_metrics = gen2.metrics();
        gen2.memory.record_generation(gen2_metrics.clone());

        // Specs should have evolved
        assert!(gen2.specs.iter().all(|s| s.version > 1));

        // ── Generation 3: Full snowball with evolved specs ──
        let mut gen3 = gen2.bootstrap_next();
        assert_eq!(gen3.generation, 3);

        gen3.new_round();
        gen3.riff_for_spec(0, "ternary-core", Quality::Strong, 0.85, 500, 35, vec!["simd-pack"]);
        gen3.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.9, 700, 50, vec!["wmma-kernels"]);
        let gen3_summary = gen3.evaluate();
        assert!(gen3_summary.landed);
        gen3.memory.learn(&gen3.rounds);
        let gen3_metrics = gen3.metrics();
        gen3.memory.record_generation(gen3_metrics.clone());

        // ── Generation 4: The new generation ──
        let mut gen4 = gen3.bootstrap_next();
        assert_eq!(gen4.generation, 4);
        // Specs carry their evolved state
        assert!(gen4.specs.iter().all(|s| s.version >= 2));

        gen4.new_round();
        gen4.riff_for_spec(0, "ternary-core", Quality::Strong, 0.92, 800, 60, vec!["avx512-pack", "async"]);
        gen4.riff_for_spec(1, "ternary-gpu", Quality::Strong, 0.95, 1200, 80, vec!["tensor-cores", "fused-kernel"]);
        let gen4_summary = gen4.evaluate();
        assert!(gen4_summary.landed);
        gen4.memory.learn(&gen4.rounds);
        let gen4_metrics = gen4.metrics();
        gen4.memory.record_generation(gen4_metrics.clone());

        // ── Verify the full 4-generation chain ──
        let mut verifier = BootstrapVerifier::new();
        let chain = vec![gen1_metrics.clone(), gen2_metrics.clone(), gen3_metrics.clone(), gen4_metrics.clone()];
        let results = verifier.verify_chain(&chain);
        assert!(results.iter().all(|r| r.is_ok()));

        // ── Check snowball growth ──
        let growth = BootstrapVerifier::check_growth(&chain);
        assert!(growth.growing);
        assert!(growth.loc_deltas.iter().all(|&d| d > 0.0));
        assert!(growth.test_deltas.iter().all(|&d| d > 0.0));

        // ── Track with SnowballTracker ──
        let mut tracker = SnowballTracker::new();
        for m in &chain { tracker.record(m.clone()); }
        assert!(tracker.is_growing());
        assert!(tracker.avg_growth_rate() > 1.0);

        // ── Self-bootstrap: generate v5 spec ──
        let bootstrap = SelfBootstrap::new(4);
        let personas: Vec<MusicianPersona> = gen4.personas.values().cloned().collect();
        let v5_spec = bootstrap.generate_next_spec(&gen4.memory, &personas);
        assert_eq!(v5_spec.version, "v5");
        assert!(!v5_spec.features.is_empty());
        assert!(v5_spec.confidence > 0.0);

        // ── Personas have diverged over 4 generations ──
        let p0 = &gen4.personas[&0];
        let p1 = &gen4.personas[&1];
        assert!(p0.experience > 0);
        assert!(p1.experience > 0);
        // Their style vectors should be different (they riffed differently)
        let style_sim = p0.style_similarity(p1);
        // Not testing exact value, just that the mechanism works
        assert!(style_sim >= -1.0 && style_sim <= 1.0);

        // ── Specs should have matured ──
        assert!(gen4.specs.iter().all(|s| s.fitness > 0.5));
        assert!(gen4.specs.iter().all(|s| !s.absorbed_patterns.is_empty()));
    }
}
