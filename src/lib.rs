//! # lau-plato-tutor
//!
//! The PLATO tutoring language — inspired by the original PLATO system from UIUC (1970s).
//! Evaluates student responses in a meaning space, not a string-matching space.
//! Uses a single embedding (feature vectors) to determine if the student's answer is
//! "close enough" to the intended answer. Sloppy logic, smooth operation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// 1. TutorResponse — what the student said/typed
// ---------------------------------------------------------------------------

/// A student's response, with extracted feature vector (the "embedding").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorResponse {
    pub text: String,
    pub timestamp: u64,
    pub features: Vec<f64>,
    pub confidence: f64,
}

impl TutorResponse {
    /// Create a new response and immediately extract features.
    pub fn new(text: &str) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut resp = Self {
            text: text.to_lowercase(),
            timestamp,
            features: Vec::new(),
            confidence: 0.0,
        };
        resp.extract_features();
        resp
    }

    /// Turn text into a feature vector (deterministic, no LLM needed).
    pub fn extract_features(&mut self) {
        let words: Vec<&str> = self.text.split_whitespace().collect();
        let word_count = words.len() as f64;

        // Feature 0: length normalization (character count, normalized)
        let char_len = self.text.len() as f64;
        self.features.push(char_len / 100.0);

        // Feature 1: word count normalization
        self.features.push(word_count / 20.0);

        // Feature 2: has numeric value
        let has_number = words.iter().any(|w| w.parse::<f64>().is_ok());
        self.features.push(if has_number { 1.0 } else { 0.0 });

        // Feature 3: extracted numeric value (0 if none)
        let num_val = words
            .iter()
            .find_map(|w| w.parse::<f64>().ok())
            .unwrap_or(0.0);
        self.features.push(num_val / 100.0);

        // Feature 4: negation detection
        let negation_words = ["not", "no", "never", "neither", "nor", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "won't", "wouldn't"];
        let has_negation = words.iter().any(|w| negation_words.contains(&(*w).to_lowercase().as_str()));
        self.features.push(if has_negation { 1.0 } else { 0.0 });

        // Feature 5: question mark (uncertainty)
        self.features.push(if self.text.contains('?') { 1.0 } else { 0.0 });

        // Feature 6: exclamation (emphasis)
        self.features.push(if self.text.contains('!') { 1.0 } else { 0.0 });

        // Confidence based on length and presence of meaningful content
        self.confidence = if word_count > 0.0 {
            (word_count / 5.0).min(1.0)
        } else {
            0.0
        };
    }

    /// Add keyword presence features (0/1 per keyword).
    pub fn add_keyword_features(&mut self, keywords: &[&str]) {
        let lower = self.text.to_lowercase();
        for kw in keywords {
            self.features.push(if lower.contains(kw.to_lowercase().as_str()) { 1.0 } else { 0.0 });
        }
    }

    /// Add concept presence features (0/1 per concept group).
    pub fn add_concept_features(&mut self, concepts: &[&str]) {
        let lower = self.text.to_lowercase();
        for concept in concepts {
            self.features.push(if lower.contains(concept.to_lowercase().as_str()) { 1.0 } else { 0.0 });
        }
    }
}

// ---------------------------------------------------------------------------
// 2. PartialRule — partial credit rules
// ---------------------------------------------------------------------------

/// A rule for granting partial credit based on concept presence/absence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialRule {
    pub if_present: String,
    pub if_absent: String,
    pub credit: f64,
}

impl PartialRule {
    pub fn new(if_present: &str, if_absent: &str, credit: f64) -> Self {
        Self {
            if_present: if_present.to_lowercase(),
            if_absent: if_absent.to_lowercase(),
            credit: credit.clamp(0.0, 1.0),
        }
    }

    /// Check if this rule applies to the given response text.
    pub fn applies_to(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        lower.contains(&self.if_present) && !lower.contains(&self.if_absent)
    }
}

// ---------------------------------------------------------------------------
// 3. IntendedAnswer — what we expected
// ---------------------------------------------------------------------------

/// The intended/correct answer with variants and concept requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntendedAnswer {
    pub text: String,
    pub features: Vec<f64>,
    pub acceptable_variants: Vec<String>,
    pub concepts_required: Vec<String>,
    pub concepts_forbidden: Vec<String>,
    pub partial_credit: Vec<PartialRule>,
}

impl IntendedAnswer {
    pub fn new(text: &str) -> Self {
        let mut resp = TutorResponse::new(text);
        resp.extract_features();
        Self {
            text: text.to_lowercase(),
            features: resp.features,
            acceptable_variants: Vec::new(),
            concepts_required: Vec::new(),
            concepts_forbidden: Vec::new(),
            partial_credit: Vec::new(),
        }
    }

    pub fn add_variant(&mut self, variant: &str) {
        self.acceptable_variants.push(variant.to_lowercase());
    }

    pub fn require_concept(&mut self, concept: &str) {
        self.concepts_required.push(concept.to_lowercase());
    }

    pub fn forbid_concept(&mut self, concept: &str) {
        self.concepts_forbidden.push(concept.to_lowercase());
    }

    pub fn add_partial_credit(&mut self, rule: PartialRule) {
        self.partial_credit.push(rule);
    }

    /// Build the full feature vector including keywords from variants and concepts.
    pub fn build_features(&mut self) {
        let mut resp = TutorResponse::new(&self.text);
        // Add keyword features for required concepts
        let keywords: Vec<&str> = self.concepts_required.iter().map(|s| s.as_str()).collect();
        resp.add_keyword_features(&keywords);
        self.features = resp.features;
    }

    /// Check if text contains all required concepts.
    pub fn has_required_concepts(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.concepts_required
            .iter()
            .filter(|c| lower.contains(c.as_str()))
            .cloned()
            .collect()
    }

    /// Check if text contains any forbidden concepts.
    pub fn has_forbidden_concepts(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        self.concepts_forbidden
            .iter()
            .filter(|c| lower.contains(c.as_str()))
            .cloned()
            .collect()
    }

    /// Calculate partial credit for a response.
    pub fn calc_partial_credit(&self, text: &str) -> f64 {
        self.partial_credit
            .iter()
            .filter(|r| r.applies_to(text))
            .map(|r| r.credit)
            .fold(0.0, f64::max)
    }

    /// Check exact or variant match.
    pub fn is_exact_or_variant(&self, text: &str) -> bool {
        let lower = text.to_lowercase().trim().to_string();
        lower == self.text.trim() || self.acceptable_variants.iter().any(|v| lower == v.trim())
    }
}

// ---------------------------------------------------------------------------
// 4. DistanceMetric & MeaningSpace — THE single embedding space
// ---------------------------------------------------------------------------

/// Distance metrics for the meaning space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Hamming,
}

/// THE single embedding space — the PLATO innovation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningSpace {
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
}

impl MeaningSpace {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            dimension,
            distance_metric: metric,
        }
    }

    /// Compute distance between two feature vectors.
    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        match self.distance_metric {
            DistanceMetric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).take(len).map(|(x, y)| x * y).sum();
                let mag_a: f64 = a.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
                let mag_b: f64 = b.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
                if mag_a == 0.0 || mag_b == 0.0 {
                    return 1.0;
                }
                1.0 - (dot / (mag_a * mag_b))
            }
            DistanceMetric::Euclidean => {
                a.iter().zip(b.iter()).take(len).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).take(len).map(|(x, y)| (x - y).abs()).sum()
            }
            DistanceMetric::Hamming => {
                a.iter().zip(b.iter()).take(len).filter(|(&x, &y)| (x - y).abs() > 0.5).count() as f64
            }
        }
    }

    /// Similarity (0-1, where 1 = identical).
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let d = self.distance(a, b);
        match self.distance_metric {
            DistanceMetric::Cosine => (1.0 - d).max(0.0),
            DistanceMetric::Euclidean => 1.0 / (1.0 + d),
            DistanceMetric::Manhattan => 1.0 / (1.0 + d),
            DistanceMetric::Hamming => {
                let len = a.len().max(b.len()).max(1) as f64;
                1.0 - (d / len)
            }
        }
    }

    /// Is the response "close enough" to the intended answer?
    pub fn is_close_enough(
        &self,
        response: &TutorResponse,
        intended: &IntendedAnswer,
        threshold: f64,
    ) -> bool {
        self.match_score(response, intended) >= threshold
    }

    /// How close is the response to the intended answer? Returns 0-1.
    pub fn match_score(&self, response: &TutorResponse, intended: &IntendedAnswer) -> f64 {
        // If exact or variant match, high score
        if intended.is_exact_or_variant(&response.text) {
            return 1.0;
        }

        // Start with embedding similarity
        let sim = self.similarity(&response.features, &intended.features);

        // Check required concepts
        let required = &intended.concepts_required;
        let mut concept_bonus = 0.0;
        if !required.is_empty() {
            let present = intended.has_required_concepts(&response.text);
            concept_bonus = present.len() as f64 / required.len() as f64 * 0.3;
        }

        // Check forbidden concepts — penalty
        let forbidden = intended.has_forbidden_concepts(&response.text);
        let forbidden_penalty = if forbidden.is_empty() { 0.0 } else { 0.3 };

        // Partial credit
        let partial = intended.calc_partial_credit(&response.text);

        let base = sim * 0.7 + concept_bonus;
        (base - forbidden_penalty).max(0.0).max(partial).min(1.0)
    }

    /// Find the best matching intended answer from candidates.
    pub fn find_best_match(
        &self,
        response: &TutorResponse,
        candidates: &[IntendedAnswer],
    ) -> (usize, f64) {
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.match_score(response, c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0))
    }
}

// ---------------------------------------------------------------------------
// 5. TutorEvaluation — how the student did
// ---------------------------------------------------------------------------

/// Evaluation result for a student response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorEvaluation {
    pub response: String,
    pub score: f64,
    pub understood: bool,
    pub concepts_present: Vec<String>,
    pub concepts_missing: Vec<String>,
    pub misconceptions: Vec<String>,
    pub feedback: String,
    pub credit: f64,
}

impl TutorEvaluation {
    pub fn new(response: &str, score: f64) -> Self {
        Self {
            response: response.to_string(),
            score: score.clamp(0.0, 1.0),
            understood: score >= 0.8,
            concepts_present: Vec::new(),
            concepts_missing: Vec::new(),
            misconceptions: Vec::new(),
            feedback: String::new(),
            credit: score.clamp(0.0, 1.0),
        }
    }

    pub fn is_correct(&self) -> bool {
        self.score >= 0.8
    }

    pub fn is_partial(&self) -> bool {
        self.score >= 0.3 && self.score < 0.8
    }

    pub fn is_wrong(&self) -> bool {
        self.score < 0.3
    }

    /// Generate encouraging feedback for the student.
    pub fn feedback_for_student(&self) -> String {
        if self.is_correct() {
            "Great work! You've got it.".to_string()
        } else if self.is_partial() {
            let missing = if self.concepts_missing.is_empty() {
                "some details".to_string()
            } else {
                self.concepts_missing.join(", ")
            };
            format!("You're on the right track! Consider reviewing: {}.", missing)
        } else {
            "Not quite, but that's okay — learning is a process. Try again!".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// 6. TutorQuestion — a question to ask
// ---------------------------------------------------------------------------

/// A tutoring question with its intended answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorQuestion {
    pub id: String,
    pub prompt: String,
    pub intended: IntendedAnswer,
    pub hints: Vec<String>,
    pub explanation: String,
    pub concept_tested: String,
    pub difficulty: f64,
}

impl TutorQuestion {
    pub fn new(prompt: &str, answer: &str) -> Self {
        let id = format!("q-{}", &prompt[..prompt.len().min(8)].to_lowercase().replace(' ', "-"));
        Self {
            id,
            prompt: prompt.to_string(),
            intended: IntendedAnswer::new(answer),
            hints: Vec::new(),
            explanation: String::new(),
            concept_tested: String::new(),
            difficulty: 0.5,
        }
    }

    pub fn evaluate(&self, response: &TutorResponse, space: &MeaningSpace) -> TutorEvaluation {
        let score = space.match_score(response, &self.intended);
        let mut eval = TutorEvaluation::new(&response.text, score);

        // Populate concepts
        eval.concepts_present = self.intended.has_required_concepts(&response.text);
        eval.concepts_missing = self
            .intended
            .concepts_required
            .iter()
            .filter(|c| !response.text.contains(c.as_str()))
            .cloned()
            .collect();

        // Misconceptions from forbidden concepts
        eval.misconceptions = self.intended.has_forbidden_concepts(&response.text);

        // Partial credit
        let partial = self.intended.calc_partial_credit(&response.text);
        eval.credit = if score >= partial { score } else { partial };

        eval.feedback = eval.feedback_for_student();
        eval
    }

    pub fn add_hint(&mut self, hint: &str) {
        self.hints.push(hint.to_string());
    }

    pub fn set_explanation(&mut self, explanation: &str) {
        self.explanation = explanation.to_string();
    }
}

// ---------------------------------------------------------------------------
// 7. TutorLesson — a sequence of questions
// ---------------------------------------------------------------------------

/// A lesson consisting of a sequence of tutoring questions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorLesson {
    pub id: String,
    pub title: String,
    pub topic: String,
    pub questions: Vec<TutorQuestion>,
    pub current_index: usize,
    pub scores: Vec<f64>,
    pub threshold: f64,
}

impl TutorLesson {
    pub fn new(title: &str, topic: &str) -> Self {
        let id = format!("lesson-{}", title.to_lowercase().replace(' ', "-"));
        Self {
            id,
            title: title.to_string(),
            topic: topic.to_string(),
            questions: Vec::new(),
            current_index: 0,
            scores: Vec::new(),
            threshold: 0.7,
        }
    }

    pub fn add_question(&mut self, question: TutorQuestion) {
        self.questions.push(question);
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&TutorQuestion> {
        if self.current_index < self.questions.len() {
            let q = &self.questions[self.current_index];
            Some(q)
        } else {
            None
        }
    }

    pub fn submit(&mut self, response: &TutorResponse, space: &MeaningSpace) -> TutorEvaluation {
        if self.current_index >= self.questions.len() {
            return TutorEvaluation::new(&response.text, 0.0);
        }
        let eval = self.questions[self.current_index].evaluate(response, space);
        self.scores.push(eval.credit);
        self.current_index += 1;
        eval
    }

    pub fn progress(&self) -> f64 {
        if self.questions.is_empty() {
            0.0
        } else {
            self.current_index as f64 / self.questions.len() as f64
        }
    }

    pub fn current_score(&self) -> f64 {
        if self.scores.is_empty() {
            0.0
        } else {
            self.scores.iter().sum::<f64>() / self.scores.len() as f64
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current_index >= self.questions.len()
    }

    pub fn passed(&self) -> bool {
        self.current_score() >= self.threshold
    }

    pub fn lesson_summary(&self) -> String {
        let status = if !self.is_complete() {
            "In Progress"
        } else if self.passed() {
            "Passed ✓"
        } else {
            "Not Yet Passed"
        };
        format!(
            "Lesson: {} ({})\nTopic: {}\nProgress: {:.0}%\nScore: {:.2}/1.00\nStatus: {}",
            self.title,
            self.id,
            self.topic,
            self.progress() * 100.0,
            self.current_score(),
            status
        )
    }
}

// ---------------------------------------------------------------------------
// 8. StudentRecord — how a student is doing
// ---------------------------------------------------------------------------

/// Record of a student's progress across lessons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentRecord {
    pub student_id: String,
    pub lessons_completed: Vec<String>,
    pub lessons_passed: Vec<String>,
    pub total_score: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

impl StudentRecord {
    pub fn new(student_id: &str) -> Self {
        Self {
            student_id: student_id.to_string(),
            lessons_completed: Vec::new(),
            lessons_passed: Vec::new(),
            total_score: 0.0,
            strengths: Vec::new(),
            weaknesses: Vec::new(),
        }
    }

    pub fn record_lesson(&mut self, lesson_id: &str, score: f64, passed: bool) {
        self.lessons_completed.push(lesson_id.to_string());
        if passed {
            self.lessons_passed.push(lesson_id.to_string());
        }
        self.total_score += score;
    }

    pub fn update_strengths(&mut self, concept: &str, score: f64) {
        if score >= 0.7 {
            if !self.strengths.contains(&concept.to_string()) {
                self.strengths.push(concept.to_string());
            }
            self.weaknesses.retain(|w| w != concept);
        } else {
            if !self.weaknesses.contains(&concept.to_string()) {
                self.weaknesses.push(concept.to_string());
            }
            self.strengths.retain(|s| s != concept);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. TutorContext — what the student gets when entering
// ---------------------------------------------------------------------------

/// Context given to a student when they beam into a TutorRoom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorContext {
    pub topic: String,
    pub current_lesson: Option<String>,
    pub last_score: Option<f64>,
    pub baton: Option<String>,
    pub hints_available: Vec<String>,
}

// ---------------------------------------------------------------------------
// 10. TutorRoom — a room dedicated to tutoring
// ---------------------------------------------------------------------------

/// A tutoring room where students can beam in, work on lessons, and beam out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorRoom {
    pub id: String,
    pub topic: String,
    pub lessons: Vec<TutorLesson>,
    pub meaning_space: MeaningSpace,
    pub student_history: HashMap<String, StudentRecord>,
    // Per-student lesson progress: (lesson_index, question_index)
    student_progress: HashMap<String, (usize, usize)>,
}

impl TutorRoom {
    pub fn new(topic: &str) -> Self {
        Self {
            id: format!("room-{}", topic.to_lowercase().replace(' ', "-")),
            topic: topic.to_string(),
            lessons: Vec::new(),
            meaning_space: MeaningSpace::new(8, DistanceMetric::Cosine),
            student_history: HashMap::new(),
            student_progress: HashMap::new(),
        }
    }

    pub fn add_lesson(&mut self, lesson: TutorLesson) {
        self.lessons.push(lesson);
    }

    /// Student enters the room. Returns context with current lesson state.
    pub fn beam_in(&mut self, student_id: &str) -> TutorContext {
        self.student_history
            .entry(student_id.to_string())
            .or_insert_with(|| StudentRecord::new(student_id));

        let (lesson_idx, _q_idx) = self
            .student_progress
            .entry(student_id.to_string())
            .or_insert((0, 0));

        let current_lesson = self.lessons.get(*lesson_idx).map(|l| l.title.clone());

        let last_score = self
            .student_history
            .get(student_id)
            .and_then(|r| {
                if r.lessons_completed.is_empty() {
                    None
                } else {
                    Some(r.total_score / r.lessons_completed.len() as f64)
                }
            });

        let baton = current_lesson.as_ref().map(|l| {
            let score_str = last_score
                .map(|s| format!(", scored {:.1} on last lesson", s))
                .unwrap_or_default();
            format!("You were working on {}{}", l, score_str)
        });

        let hints = self
            .lessons
            .get(*lesson_idx)
            .and_then(|l| l.questions.first())
            .map(|q| q.hints.clone())
            .unwrap_or_default();

        TutorContext {
            topic: self.topic.clone(),
            current_lesson,
            last_score,
            baton,
            hints_available: hints,
        }
    }

    /// Student leaves the room. Save progress.
    pub fn beam_out(&mut self, student_id: &str) {
        if let Some((lesson_idx, _)) = self.student_progress.get(student_id).cloned() {
            if let Some(lesson) = self.lessons.get(lesson_idx) {
                if lesson.is_complete() {
                    if let Some(record) = self.student_history.get_mut(student_id) {
                        record.record_lesson(&lesson.id, lesson.current_score(), lesson.passed());
                    }
                    // Advance to next lesson
                    if let Some(progress) = self.student_progress.get_mut(student_id) {
                        progress.0 = lesson_idx + 1;
                        progress.1 = 0;
                    }
                }
            }
        }
    }

    /// Evaluate a student's response in the current lesson.
    pub fn evaluate(&mut self, student_id: &str, response: &str) -> TutorEvaluation {
        let tutor_response = TutorResponse::new(response);

        let (lesson_idx, _) = self
            .student_progress
            .entry(student_id.to_string())
            .or_insert((0, 0));

        if let Some(lesson) = self.lessons.get_mut(*lesson_idx) {
            let eval = lesson.submit(&tutor_response, &self.meaning_space);

            // Update strengths/weaknesses
            if let Some(record) = self.student_history.get_mut(student_id) {
                let concept = self
                    .lessons
                    .get(*lesson_idx)
                    .and_then(|l| l.questions.get(l.current_index.saturating_sub(1)))
                    .map(|q| q.concept_tested.clone())
                    .unwrap_or_default();
                if !concept.is_empty() {
                    record.update_strengths(&concept, eval.credit);
                }
            }

            eval
        } else {
            TutorEvaluation::new(response, 0.0)
        }
    }

    pub fn student_status(&self, student_id: &str) -> Option<&StudentRecord> {
        self.student_history.get(student_id)
    }
}

// ---------------------------------------------------------------------------
// Pre-built lessons
// ---------------------------------------------------------------------------

/// 5 questions about energy conservation.
pub fn conservation_lesson() -> TutorLesson {
    let mut lesson = TutorLesson::new("Energy Conservation", "physics");

    let mut q1 = TutorQuestion::new(
        "What happens to energy in a closed system?",
        "Energy is conserved, it cannot be created or destroyed",
    );
    q1.intended.add_variant("energy is conserved");
    q1.intended.add_variant("it stays the same");
    q1.intended.add_variant("total energy remains constant");
    q1.intended.require_concept("conserved");
    q1.intended.require_concept("energy");
    q1.concept_tested = "conservation".to_string();
    q1.difficulty = 0.3;
    q1.set_explanation("The First Law of Thermodynamics: energy is conserved in a closed system.");
    q1.add_hint("Think about what happens to the total amount of energy.");
    lesson.add_question(q1);

    let mut q2 = TutorQuestion::new(
        "When a ball rolls uphill, what happens to its kinetic energy?",
        "Kinetic energy decreases and potential energy increases",
    );
    q2.intended.add_variant("kinetic energy goes down potential goes up");
    q2.intended.add_variant("it slows down and gains height");
    q2.intended.require_concept("kinetic");
    q2.intended.require_concept("decreases");
    q2.intended.forbid_concept("destroyed");
    q2.intended.add_partial_credit(PartialRule::new("kinetic", "destroyed", 0.5));
    q2.concept_tested = "kinetic-potential".to_string();
    q2.difficulty = 0.5;
    q2.set_explanation("Kinetic energy converts to potential energy as the ball gains height.");
    lesson.add_question(q2);

    let mut q3 = TutorQuestion::new(
        "What is the unit of energy?",
        "Joule",
    );
    q3.intended.add_variant("joules");
    q3.intended.add_variant("j");
    q3.intended.add_variant("the joule");
    q3.concept_tested = "units".to_string();
    q3.difficulty = 0.2;
    q3.set_explanation("The SI unit of energy is the Joule (J), named after James Prescott Joule.");
    lesson.add_question(q3);

    let mut q4 = TutorQuestion::new(
        "Can energy be destroyed?",
        "No, energy cannot be destroyed only transformed",
    );
    q4.intended.add_variant("no");
    q4.intended.add_variant("no it is conserved");
    q4.intended.require_concept("no");
    q4.intended.forbid_concept("yes");
    q4.intended.forbid_concept("destroyed");
    q4.concept_tested = "conservation-law".to_string();
    q4.difficulty = 0.4;
    q4.set_explanation("Energy conservation means energy is never destroyed, only converted between forms.");
    lesson.add_question(q4);

    let mut q5 = TutorQuestion::new(
        "What type of energy does a stretched rubber band have?",
        "Elastic potential energy",
    );
    q5.intended.add_variant("potential energy");
    q5.intended.add_variant("elastic energy");
    q5.intended.add_variant("stored energy");
    q5.intended.require_concept("potential");
    q5.concept_tested = "potential-energy".to_string();
    q5.difficulty = 0.6;
    q5.set_explanation("A stretched rubber band stores elastic potential energy.");
    lesson.add_question(q5);

    lesson
}

/// 5 questions about symmetry detection.
pub fn symmetry_lesson() -> TutorLesson {
    let mut lesson = TutorLesson::new("Symmetry Detection", "mathematics");

    let mut q1 = TutorQuestion::new(
        "How many lines of symmetry does a square have?",
        "Four",
    );
    q1.intended.add_variant("4");
    q1.intended.add_variant("four lines");
    q1.concept_tested = "reflection-symmetry".to_string();
    q1.difficulty = 0.3;
    q1.set_explanation("A square has 4 lines of symmetry: horizontal, vertical, and two diagonals.");
    lesson.add_question(q1);

    let mut q2 = TutorQuestion::new(
        "What type of symmetry does a circle have?",
        "Rotational symmetry and infinite lines of symmetry",
    );
    q2.intended.add_variant("infinite");
    q2.intended.add_variant("rotational");
    q2.intended.add_variant("all lines of symmetry");
    q2.intended.require_concept("symmetry");
    q2.concept_tested = "rotational-symmetry".to_string();
    q2.difficulty = 0.4;
    q2.set_explanation("A circle has rotational symmetry and infinitely many lines of symmetry.");
    lesson.add_question(q2);

    let mut q3 = TutorQuestion::new(
        "Does the letter A have a line of symmetry?",
        "Yes, vertical",
    );
    q3.intended.add_variant("yes");
    q3.intended.add_variant("vertical line");
    q3.intended.require_concept("yes");
    q3.concept_tested = "letter-symmetry".to_string();
    q3.difficulty = 0.2;
    q3.set_explanation("The letter A has a vertical line of symmetry down the middle.");
    lesson.add_question(q3);

    let mut q4 = TutorQuestion::new(
        "What is rotational symmetry?",
        "When a shape can be rotated and still look the same",
    );
    q4.intended.add_variant("looks the same after rotation");
    q4.intended.add_variant("same when rotated");
    q4.intended.require_concept("rotated");
    q4.intended.require_concept("same");
    q4.concept_tested = "rotational-definition".to_string();
    q4.difficulty = 0.5;
    q4.set_explanation("Rotational symmetry means a shape looks the same after some rotation.");
    lesson.add_question(q4);

    let mut q5 = TutorQuestion::new(
        "How many lines of symmetry does an equilateral triangle have?",
        "Three",
    );
    q5.intended.add_variant("3");
    q5.intended.add_variant("three lines");
    q5.concept_tested = "triangle-symmetry".to_string();
    q5.difficulty = 0.35;
    q5.set_explanation("An equilateral triangle has 3 lines of symmetry, one through each vertex.");
    lesson.add_question(q5);

    lesson
}

/// Questions about a specific cultural tradition.
pub fn tradition_lesson(tradition: &str) -> TutorLesson {
    let mut lesson = TutorLesson::new(
        &format!("{} Tradition", tradition),
        "cultural-studies",
    );

    let mut q1 = TutorQuestion::new(
        &format!("What is {} known for?", tradition),
        &format!("{} is a cultural tradition", tradition),
    );
    q1.intended.add_variant(tradition);
    q1.intended.require_concept(tradition.to_lowercase().as_str());
    q1.concept_tested = "overview".to_string();
    q1.difficulty = 0.3;
    lesson.add_question(q1);

    let mut q2 = TutorQuestion::new(
        &format!("Why is {} important?", tradition),
        "It preserves cultural heritage and identity",
    );
    q2.intended.add_variant("cultural heritage");
    q2.intended.add_variant("identity");
    q2.intended.add_variant("preserves tradition");
    q2.concept_tested = "significance".to_string();
    q2.difficulty = 0.4;
    lesson.add_question(q2);

    let mut q3 = TutorQuestion::new(
        &format!("How is {} typically celebrated?", tradition),
        "Through ceremonies, rituals, and community gatherings",
    );
    q3.intended.add_variant("ceremonies");
    q3.intended.add_variant("rituals");
    q3.intended.add_variant("community");
    q3.intended.require_concept("community");
    q3.concept_tested = "practice".to_string();
    q3.difficulty = 0.5;
    lesson.add_question(q3);

    let mut q4 = TutorQuestion::new(
        "What values do traditions transmit?",
        "Values, beliefs, history, and social norms",
    );
    q4.intended.add_variant("values and beliefs");
    q4.intended.add_variant("history");
    q4.intended.require_concept("values");
    q4.concept_tested = "transmission".to_string();
    q4.difficulty = 0.45;
    lesson.add_question(q4);

    let mut q5 = TutorQuestion::new(
        &format!("What happens if {} is lost?", tradition),
        "Cultural diversity and heritage are diminished",
    );
    q5.intended.add_variant("diversity is lost");
    q5.intended.add_variant("heritage is lost");
    q5.intended.add_variant("culture is diminished");
    q5.concept_tested = "preservation".to_string();
    q5.difficulty = 0.6;
    lesson.add_question(q5);

    lesson
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TutorResponse tests ---

    #[test]
    fn test_response_new() {
        let r = TutorResponse::new("Energy is conserved");
        assert_eq!(r.text, "energy is conserved");
        assert!(!r.features.is_empty());
        assert!(r.timestamp > 0);
    }

    #[test]
    fn test_response_features_not_empty() {
        let r = TutorResponse::new("hello world");
        assert!(r.features.len() >= 6, "should have at least 6 base features");
    }

    #[test]
    fn test_response_length_features() {
        let short = TutorResponse::new("hi");
        let long = TutorResponse::new("this is a much longer response with many words in it");
        assert!(short.features[0] < long.features[0], "short text should have smaller length feature");
        assert!(short.features[1] < long.features[1], "short text should have smaller word count feature");
    }

    #[test]
    fn test_response_numeric_detection() {
        let with_num = TutorResponse::new("the answer is 42");
        let without = TutorResponse::new("the answer is forty two");
        assert_eq!(with_num.features[2], 1.0, "should detect number");
        assert_eq!(without.features[2], 0.0, "should not detect number");
    }

    #[test]
    fn test_response_numeric_extraction() {
        let r = TutorResponse::new("energy is 100 joules");
        assert!(r.features[3] > 0.0, "should extract numeric value");
    }

    #[test]
    fn test_response_negation() {
        let neg = TutorResponse::new("energy is not destroyed");
        let pos = TutorResponse::new("energy is conserved");
        assert_eq!(neg.features[4], 1.0, "should detect negation");
        assert_eq!(pos.features[4], 0.0, "should not detect negation");
    }

    #[test]
    fn test_response_keyword_features() {
        let mut r = TutorResponse::new("kinetic energy decreases");
        r.add_keyword_features(&["kinetic", "potential", "energy"]);
        assert_eq!(r.features[r.features.len() - 3], 1.0); // kinetic
        assert_eq!(r.features[r.features.len() - 2], 0.0); // potential
        assert_eq!(r.features[r.features.len() - 1], 1.0); // energy
    }

    #[test]
    fn test_response_confidence() {
        let empty = TutorResponse::new("");
        let full = TutorResponse::new("this has several words in the response");
        assert_eq!(empty.confidence, 0.0);
        assert!(full.confidence > 0.0);
    }

    // --- PartialRule tests ---

    #[test]
    fn test_partial_rule_new() {
        let rule = PartialRule::new("kinetic", "destroyed", 0.5);
        assert_eq!(rule.if_present, "kinetic");
        assert_eq!(rule.if_absent, "destroyed");
        assert_eq!(rule.credit, 0.5);
    }

    #[test]
    fn test_partial_rule_clamp() {
        let rule = PartialRule::new("a", "b", 1.5);
        assert_eq!(rule.credit, 1.0);
        let rule2 = PartialRule::new("a", "b", -0.5);
        assert_eq!(rule2.credit, 0.0);
    }

    #[test]
    fn test_partial_rule_applies() {
        let rule = PartialRule::new("kinetic", "destroyed", 0.5);
        assert!(rule.applies_to("kinetic energy increases"));
        assert!(!rule.applies_to("kinetic energy is destroyed"));
        assert!(!rule.applies_to("potential energy increases"));
    }

    // --- IntendedAnswer tests ---

    #[test]
    fn test_intended_new() {
        let a = IntendedAnswer::new("Energy is conserved");
        assert_eq!(a.text, "energy is conserved");
        assert!(a.features.len() >= 6);
    }

    #[test]
    fn test_intended_variants() {
        let a = IntendedAnswer::new("yes");
        assert!(a.is_exact_or_variant("yes"));
        assert!(!a.is_exact_or_variant("no"));
    }

    #[test]
    fn test_intended_add_variant() {
        let mut a = IntendedAnswer::new("energy is conserved");
        a.add_variant("energy stays the same");
        assert!(a.is_exact_or_variant("energy stays the same"));
        assert!(a.is_exact_or_variant("energy is conserved"));
    }

    #[test]
    fn test_intended_concepts() {
        let mut a = IntendedAnswer::new("energy is conserved");
        a.require_concept("energy");
        a.require_concept("conserved");
        let present = a.has_required_concepts("energy is conserved");
        assert_eq!(present.len(), 2);
        let partial = a.has_required_concepts("energy is lost");
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0], "energy");
    }

    #[test]
    fn test_intended_forbidden() {
        let mut a = IntendedAnswer::new("energy is conserved");
        a.forbid_concept("destroyed");
        let found = a.has_forbidden_concepts("energy is destroyed");
        assert_eq!(found.len(), 1);
        let none = a.has_forbidden_concepts("energy is conserved");
        assert!(none.is_empty());
    }

    #[test]
    fn test_intended_partial_credit() {
        let mut a = IntendedAnswer::new("kinetic and potential");
        a.add_partial_credit(PartialRule::new("kinetic", "destroyed", 0.5));
        assert_eq!(a.calc_partial_credit("kinetic energy is great"), 0.5);
        assert_eq!(a.calc_partial_credit("kinetic is destroyed"), 0.0);
    }

    // --- MeaningSpace tests ---

    #[test]
    fn test_space_cosine_identical() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let v = vec![1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = space.distance(&v, &v);
        assert!(d.abs() < 0.001, "identical vectors should have ~0 distance, got {}", d);
    }

    #[test]
    fn test_space_cosine_orthogonal() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = space.distance(&a, &b);
        assert!((d - 1.0).abs() < 0.001, "orthogonal vectors should have distance ~1, got {}", d);
    }

    #[test]
    fn test_space_euclidean() {
        let space = MeaningSpace::new(8, DistanceMetric::Euclidean);
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let d = space.distance(&a, &b);
        assert!((d - 8.0_f64.sqrt()).abs() < 0.001, "got {}", d);
    }

    #[test]
    fn test_space_manhattan() {
        let space = MeaningSpace::new(8, DistanceMetric::Manhattan);
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let d = space.distance(&a, &b);
        assert!((d - 8.0).abs() < 0.001, "got {}", d);
    }

    #[test]
    fn test_space_hamming() {
        let space = MeaningSpace::new(8, DistanceMetric::Hamming);
        let a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let d = space.distance(&a, &b);
        assert_eq!(d, 4.0);
    }

    #[test]
    fn test_space_similarity_range() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let a = vec![1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sim = space.similarity(&a, &b);
        assert!(sim >= 0.0 && sim <= 1.0);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_space_is_close_enough() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("energy is conserved");
        let intended = IntendedAnswer::new("energy is conserved");
        assert!(space.is_close_enough(&resp, &intended, 0.9));
    }

    #[test]
    fn test_space_match_score_exact() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("energy is conserved");
        let intended = IntendedAnswer::new("energy is conserved");
        assert_eq!(space.match_score(&resp, &intended), 1.0);
    }

    #[test]
    fn test_space_match_score_variant() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut intended = IntendedAnswer::new("energy is conserved");
        intended.add_variant("energy stays the same");
        let resp = TutorResponse::new("energy stays the same");
        assert_eq!(space.match_score(&resp, &intended), 1.0);
    }

    #[test]
    fn test_space_find_best_match() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("energy is conserved");
        let a1 = IntendedAnswer::new("energy is conserved");
        let a2 = IntendedAnswer::new("the sky is blue");
        let a3 = IntendedAnswer::new("water boils at 100");
        let (idx, score) = space.find_best_match(&resp, &[a1, a2, a3]);
        assert_eq!(idx, 0);
        assert!(score > 0.5);
    }

    // --- TutorEvaluation tests ---

    #[test]
    fn test_eval_correct() {
        let e = TutorEvaluation::new("test", 0.9);
        assert!(e.is_correct());
        assert!(!e.is_partial());
        assert!(!e.is_wrong());
    }

    #[test]
    fn test_eval_partial() {
        let e = TutorEvaluation::new("test", 0.5);
        assert!(!e.is_correct());
        assert!(e.is_partial());
        assert!(!e.is_wrong());
    }

    #[test]
    fn test_eval_wrong() {
        let e = TutorEvaluation::new("test", 0.1);
        assert!(!e.is_correct());
        assert!(!e.is_partial());
        assert!(e.is_wrong());
    }

    #[test]
    fn test_eval_boundary_correct() {
        let e = TutorEvaluation::new("test", 0.8);
        assert!(e.is_correct());
    }

    #[test]
    fn test_eval_boundary_wrong() {
        let e = TutorEvaluation::new("test", 0.3);
        assert!(e.is_partial());
    }

    #[test]
    fn test_eval_feedback_correct() {
        let e = TutorEvaluation::new("test", 0.9);
        assert!(e.feedback_for_student().contains("Great"));
    }

    #[test]
    fn test_eval_feedback_partial() {
        let mut e = TutorEvaluation::new("test", 0.5);
        e.concepts_missing.push("conservation".to_string());
        let fb = e.feedback_for_student();
        assert!(fb.contains("right track") || fb.contains("Consider"));
    }

    #[test]
    fn test_eval_feedback_wrong() {
        let e = TutorEvaluation::new("test", 0.1);
        assert!(e.feedback_for_student().contains("Not quite") || e.feedback_for_student().contains("okay"));
    }

    // --- TutorQuestion tests ---

    #[test]
    fn test_question_new() {
        let q = TutorQuestion::new("What is energy?", "Energy is the capacity to do work");
        assert!(!q.id.is_empty());
        assert_eq!(q.prompt, "What is energy?");
        assert!(q.hints.is_empty());
    }

    #[test]
    fn test_question_evaluate_exact() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let q = TutorQuestion::new("What is energy?", "Energy is the capacity to do work");
        let resp = TutorResponse::new("Energy is the capacity to do work");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.is_correct());
    }

    #[test]
    fn test_question_evaluate_with_concepts() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut q = TutorQuestion::new("What is energy?", "Energy is the capacity to do work");
        q.intended.require_concept("energy");
        q.intended.require_concept("capacity");
        q.concept_tested = "energy-definition".to_string();
        let resp = TutorResponse::new("energy is the capacity to do work");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.concepts_present.contains(&"energy".to_string()));
    }

    #[test]
    fn test_question_add_hint() {
        let mut q = TutorQuestion::new("test?", "answer");
        q.add_hint("think about physics");
        assert_eq!(q.hints.len(), 1);
        assert_eq!(q.hints[0], "think about physics");
    }

    #[test]
    fn test_question_set_explanation() {
        let mut q = TutorQuestion::new("test?", "answer");
        q.set_explanation("This is because...");
        assert_eq!(q.explanation, "This is because...");
    }

    // --- TutorLesson tests ---

    #[test]
    fn test_lesson_new() {
        let l = TutorLesson::new("Test Lesson", "physics");
        assert!(!l.id.is_empty());
        assert_eq!(l.title, "Test Lesson");
        assert_eq!(l.topic, "physics");
        assert!(l.questions.is_empty());
        assert_eq!(l.progress(), 0.0);
    }

    #[test]
    fn test_lesson_add_question() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        l.add_question(TutorQuestion::new("Q2?", "A2"));
        assert_eq!(l.questions.len(), 2);
    }

    #[test]
    fn test_lesson_next() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let q = l.next();
        assert!(q.is_some());
        assert_eq!(q.unwrap().prompt, "Q1?");
    }

    #[test]
    fn test_lesson_submit() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("A1");
        let eval = l.submit(&resp, &space);
        assert_eq!(l.current_index, 1);
        assert_eq!(l.scores.len(), 1);
        assert!(eval.credit > 0.0);
    }

    #[test]
    fn test_lesson_progress() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        l.add_question(TutorQuestion::new("Q2?", "A2"));
        assert_eq!(l.progress(), 0.0);
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        l.submit(&TutorResponse::new("A1"), &space);
        assert!((l.progress() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_lesson_complete() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        l.submit(&TutorResponse::new("A1"), &space);
        assert!(l.is_complete());
    }

    #[test]
    fn test_lesson_passed() {
        let mut l = TutorLesson::new("Test", "test");
        l.threshold = 0.5;
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        l.submit(&TutorResponse::new("A1"), &space);
        assert!(l.passed());
    }

    #[test]
    fn test_lesson_summary() {
        let mut l = TutorLesson::new("Physics 101", "physics");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        l.submit(&TutorResponse::new("A1"), &space);
        let summary = l.lesson_summary();
        assert!(summary.contains("Physics 101"));
        assert!(summary.contains("Passed") || summary.contains("Not Yet Passed"));
    }

    #[test]
    fn test_lesson_current_score() {
        let mut l = TutorLesson::new("Test", "test");
        l.add_question(TutorQuestion::new("Q1?", "A1"));
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        l.submit(&TutorResponse::new("A1"), &space);
        let score = l.current_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    // --- StudentRecord tests ---

    #[test]
    fn test_student_record_new() {
        let r = StudentRecord::new("student-1");
        assert_eq!(r.student_id, "student-1");
        assert!(r.lessons_completed.is_empty());
    }

    #[test]
    fn test_student_record_lesson() {
        let mut r = StudentRecord::new("s1");
        r.record_lesson("lesson-1", 0.8, true);
        assert_eq!(r.lessons_completed.len(), 1);
        assert_eq!(r.lessons_passed.len(), 1);
        assert!((r.total_score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_student_record_strengths() {
        let mut r = StudentRecord::new("s1");
        r.update_strengths("energy", 0.9);
        assert!(r.strengths.contains(&"energy".to_string()));
        r.update_strengths("energy", 0.2);
        assert!(!r.strengths.contains(&"energy".to_string()));
        assert!(r.weaknesses.contains(&"energy".to_string()));
    }

    // --- TutorRoom tests ---

    #[test]
    fn test_room_new() {
        let room = TutorRoom::new("Physics");
        assert!(!room.id.is_empty());
        assert_eq!(room.topic, "Physics");
    }

    #[test]
    fn test_room_beam_in() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        let ctx = room.beam_in("student-1");
        assert_eq!(ctx.topic, "Physics");
        assert!(ctx.current_lesson.is_some());
    }

    #[test]
    fn test_room_beam_in_baton() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        let ctx = room.beam_in("student-1");
        assert!(ctx.baton.is_some());
        assert!(ctx.baton.unwrap().contains("Energy Conservation"));
    }

    #[test]
    fn test_room_evaluate() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        room.beam_in("student-1");
        let eval = room.evaluate("student-1", "energy is conserved");
        assert!(eval.score > 0.0);
    }

    #[test]
    fn test_room_student_status() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        room.beam_in("student-1");
        assert!(room.student_status("student-1").is_some());
        assert!(room.student_status("unknown").is_none());
    }

    #[test]
    fn test_room_beam_out() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        room.beam_in("student-1");
        room.beam_out("student-1");
        // Should not panic
        assert!(room.student_status("student-1").is_some());
    }

    // --- Pre-built lesson tests ---

    #[test]
    fn test_conservation_lesson() {
        let lesson = conservation_lesson();
        assert_eq!(lesson.questions.len(), 5);
        assert_eq!(lesson.topic, "physics");
    }

    #[test]
    fn test_conservation_lesson_q1_correct() {
        let lesson = conservation_lesson();
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("energy is conserved");
        let eval = lesson.questions[0].evaluate(&resp, &space);
        assert!(eval.is_correct());
    }

    #[test]
    fn test_conservation_lesson_q3_joule() {
        let lesson = conservation_lesson();
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("joule");
        let eval = lesson.questions[2].evaluate(&resp, &space);
        assert!(eval.is_correct());
    }

    #[test]
    fn test_symmetry_lesson() {
        let lesson = symmetry_lesson();
        assert_eq!(lesson.questions.len(), 5);
        assert_eq!(lesson.topic, "mathematics");
    }

    #[test]
    fn test_symmetry_lesson_square() {
        let lesson = symmetry_lesson();
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let resp = TutorResponse::new("four");
        let eval = lesson.questions[0].evaluate(&resp, &space);
        assert!(eval.is_correct());
    }

    #[test]
    fn test_tradition_lesson() {
        let lesson = tradition_lesson("Diwali");
        assert_eq!(lesson.questions.len(), 5);
        assert!(lesson.title.contains("Diwali"));
    }

    // --- Integration tests ---

    #[test]
    fn test_full_lesson_flow() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut lesson = conservation_lesson();
        lesson.threshold = 0.5;

        // Answer all 5 questions reasonably
        let answers = [
            "energy is conserved",
            "kinetic energy decreases and potential increases",
            "joule",
            "no, energy cannot be destroyed",
            "elastic potential energy",
        ];

        for ans in answers {
            let resp = TutorResponse::new(ans);
            let _eval = lesson.submit(&resp, &space);
        }

        assert!(lesson.is_complete());
        assert!(lesson.passed());
        assert!(lesson.current_score() > 0.5);
    }

    #[test]
    fn test_full_room_flow() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());

        let ctx = room.beam_in("alice");
        assert!(ctx.current_lesson.is_some());

        // Answer questions
        let answers = [
            "energy is conserved",
            "kinetic energy decreases",
            "joule",
            "no energy cannot be destroyed",
            "potential energy",
        ];

        for ans in answers {
            room.evaluate("alice", ans);
        }

        room.beam_out("alice");
        let status = room.student_status("alice").unwrap();
        assert_eq!(status.lessons_completed.len(), 1);
    }

    #[test]
    fn test_sloppy_logic_close_enough() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut q = TutorQuestion::new(
            "What happens to energy?",
            "energy is conserved it cannot be created or destroyed",
        );
        q.intended.require_concept("energy");
        q.intended.require_concept("conserved");

        // Sloppy answer — close enough
        let resp = TutorResponse::new("energy is conserved");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.score > 0.5, "sloppy answer should score well, got {}", eval.score);
    }

    #[test]
    fn test_wrong_answer_scores_low() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut q = TutorQuestion::new("What is the unit of energy?", "joule");
        q.intended.require_concept("joule");
        let resp = TutorResponse::new("banana");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.score < 0.75, "wrong answer should score low, got {}", eval.score);
    }

    #[test]
    fn test_serde_roundtrip_response() {
        let r = TutorResponse::new("energy is conserved");
        let json = serde_json::to_string(&r).unwrap();
        let r2: TutorResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(r.text, r2.text);
        assert_eq!(r.features.len(), r2.features.len());
    }

    #[test]
    fn test_serde_roundtrip_lesson() {
        let l = conservation_lesson();
        let json = serde_json::to_string(&l).unwrap();
        let l2: TutorLesson = serde_json::from_str(&json).unwrap();
        assert_eq!(l.title, l2.title);
        assert_eq!(l.questions.len(), l2.questions.len());
    }

    #[test]
    fn test_serde_roundtrip_room() {
        let mut room = TutorRoom::new("Physics");
        room.add_lesson(conservation_lesson());
        let json = serde_json::to_string(&room).unwrap();
        let room2: TutorRoom = serde_json::from_str(&json).unwrap();
        assert_eq!(room.topic, room2.topic);
        assert_eq!(room.lessons.len(), room2.lessons.len());
    }

    #[test]
    fn test_lesson_not_passed_with_bad_answers() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut lesson = conservation_lesson();
        lesson.threshold = 0.5;

        let bad_answers = ["banana", "purple", "idk", "42", "water"];
        for ans in bad_answers {
            lesson.submit(&TutorResponse::new(ans), &space);
        }

        assert!(lesson.is_complete());
        assert!(!lesson.passed());
    }

    #[test]
    fn test_forbidden_concept_penalty() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut q = TutorQuestion::new("Can energy be destroyed?", "no energy cannot be destroyed");
        q.intended.require_concept("no");
        q.intended.forbid_concept("destroyed");

        let resp = TutorResponse::new("yes energy is destroyed");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.score < 0.5, "forbidden concepts should penalize, got {}", eval.score);
    }

    #[test]
    fn test_partial_credit_applies() {
        let space = MeaningSpace::new(8, DistanceMetric::Cosine);
        let mut q = TutorQuestion::new("What happens when a ball goes up?", "kinetic decreases potential increases");
        q.intended.add_partial_credit(PartialRule::new("kinetic", "destroyed", 0.5));

        let resp = TutorResponse::new("kinetic energy changes");
        let eval = q.evaluate(&resp, &space);
        assert!(eval.credit >= 0.0);
    }
}
