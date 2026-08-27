fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0x3040..=0x30FF | 0xF900..=0xFAFF)
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                if is_cjk(lc) {
                    if !word.is_empty() {
                        out.push(std::mem::take(&mut word));
                    }
                    out.push(lc.to_string());
                } else {
                    word.push(lc);
                }
            }
        } else if !word.is_empty() {
            out.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

pub fn char_trigrams(word: &str) -> Vec<String> {
    let marked: String = format!("\u{1}<{}>\u{1}", word);
    let chars: Vec<char> = marked.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }
    let cap = chars.len().min(64);
    let mut out = Vec::with_capacity(cap);
    for w in chars[..cap].windows(3) {
        out.push(w.iter().collect::<String>());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_and_cjk() {
        let toks = tokenize("Hello, 世界! user_id");
        assert_eq!(
            toks,
            vec!["hello", "世", "界", "user", "id"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trigrams_have_boundaries() {
        let g = char_trigrams("abc");
        assert_eq!(g.len(), 5);
        assert!(g[0].contains('<'));
    }

    #[test]
    fn empty_input() {
        assert!(tokenize(" !!! --- ").is_empty());
    }
}

