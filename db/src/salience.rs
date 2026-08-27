#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Hot,
    Warm,
    Cold,
}

pub const W_RECENCY: f32 = 0.5;
pub const W_IMPORTANCE: f32 = 0.3;
pub const W_ACCESS: f32 = 0.2;

pub fn recency(now: i64, event_time: i64, half_life_secs: f64) -> f32 {
    let dt = (now - event_time).max(0) as f64;
    (-dt / half_life_secs * std::f64::consts::LN_2).exp() as f32
}

pub fn access_boost(hits: u32) -> f32 {
    ((hits as f32).ln_1p()).min(3.0) / 3.0
}

pub fn salience(rec: &crate::types::Record, now: i64, half_life_secs: f64) -> f32 {
    (W_RECENCY * recency(now, rec.event_time, half_life_secs)
        + W_IMPORTANCE * rec.importance
        + W_ACCESS * access_boost(rec.hits))
    .clamp(0.0, 1.0)
}

pub fn tier(salience: f32) -> Tier {
    if salience > 0.66 {
        Tier::Hot
    } else if salience > 0.33 {
        Tier::Warm
    } else {
        Tier::Cold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Record;

    fn rec(event_time: i64, importance: f32, hits: u32) -> Record {
        Record {
            id: 1,
            text: "t".into(),
            subject: None,
            tags: vec![],
            event_time,
            ingest_time: event_time,
            valid_to: None,
            importance,
            hits,
            last_hit: None,
            source_ids: Vec::new(),
        }
    }

    #[test]
    fn fresh_important_is_hot() {
        let s = salience(&rec(1000, 0.9, 0), 1000, 86400.0);
        assert_eq!(tier(s), Tier::Hot);
    }

    #[test]
    fn old_forgotten_is_cold() {
        let s = salience(&rec(0, 0.1, 0), 86_400 * 30, 86400.0);
        assert_eq!(tier(s), Tier::Cold);
    }

    #[test]
    fn recency_halves_each_half_life() {
        let r0 = recency(1000, 1000, 100.0);
        let r1 = recency(1100, 1000, 100.0);
        assert!((r0 - 1.0).abs() < 1e-6);
        assert!((r1 - 0.5).abs() < 1e-3, "{r1}");
    }
}

