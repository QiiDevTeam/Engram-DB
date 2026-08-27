use engram_db::collection::RecallOpts;
use engram_db::db::Db;
use engram_db::{Profile, RememberOpts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("engram-quickstart");
    let _ = std::fs::remove_dir_all(&dir);

    let db = Db::open(&dir)?;
    let col = db.create_collection("assistant")?;

    col.remember(RememberOpts::new("用户住在上海，最喜欢的城市是杭州"))?;
    col.remember(
        RememberOpts::new("用户的生日是3月14日")
            .subject("user.birthday")
            .importance(0.9),
    )?;
    col.remember(RememberOpts::new("用户正在用 Rust 写一个向量数据库项目").importance(0.8))?;
    db.checkpoint_all()?;

    let hits = col
        .recall(
            RecallOpts::new("用户对什么感兴趣？")
                .budget_tokens(300)
                .profile(Profile::Chat),
        )
        .unwrap();

    println!("recall results (budget=300 tokens):");
    for h in hits {
        println!(
            "  [{:.3}] (sem={:.2} lex={:.2} rec={:.2}) ~{}tok  {}",
            h.score, h.semantic, h.lexical, h.recency, h.estimated_tokens, h.text
        );
    }

    let updated = col.remember(RememberOpts::new("用户搬到了深圳").subject("user.city"))?;
    let (live, total) = col.stats();
    println!("\nafter subject supersede: live={live} total={total} new_id={updated}");

    Ok(())
}

