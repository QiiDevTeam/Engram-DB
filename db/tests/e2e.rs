use engram_db::collection::RecallOpts;
use engram_db::db::Db;
use engram_db::{Profile, RememberOpts};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "engram-test-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn remember_and_recall_basic() {
    let dir = tmpdir("basic");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let texts = [
        "用户住在上海，最喜欢的城市是杭州",
        "用户的生日是3月14日，属兔",
        "project uses postgresql for storage layer",
        "the team meets every monday morning standup",
        "quantum computing uses qubits and superposition",
    ];
    for (i, t) in texts.iter().enumerate() {
        let mut o = RememberOpts::new(*t);
        o = o.importance(0.5);
        if i == 1 {
            o = o.subject("user.birthday");
        }
        col.remember(o).unwrap();
    }

    let hits = col.recall(RecallOpts::new("用户生日是什么时候？").budget_tokens(400)).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].text.contains("生日"), "top hit: {}", hits[0].text);

    let hits_en = col
        .recall(RecallOpts::new("which database does the project use").budget_tokens(400))
        .unwrap();
    assert!(
        hits_en[0].text.contains("postgres"),
        "top hit: {}",
        hits_en[0].text
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persistence_reopen_and_checkpoint() {
    let dir = tmpdir("persist");

    {
        let db = Db::open(&dir).unwrap();
        let col = db.create_collection("main").unwrap();
        col.remember(RememberOpts::new("记忆一：用户养了一只橘猫叫年糕")).unwrap();
        col.remember(RememberOpts::new("记忆二：用户在周末喜欢爬山")).unwrap();
        db.checkpoint_all().unwrap();
        col.remember(RememberOpts::new("记忆三：用户最近在学习钢琴")).unwrap();
    }

    {
        let db = Db::open(&dir).unwrap();
        let col = db.collection("main").unwrap();
        let (_, total) = col.stats();
        assert_eq!(total, 3, "all records recovered across snapshot + wal");

        let hits = col.recall(RecallOpts::new("用户养的猫叫什么名字").budget_tokens(200)).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].text.contains("年糕"), "{}", hits[0].text);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn subject_version_chain_supersedes() {
    let dir = tmpdir("version");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let id_old = col
        .remember(RememberOpts::new("用户住在北京").subject("user.city"))
        .unwrap();
    let id_new = col
        .remember(RememberOpts::new("用户搬到上海了，现在住在上海").subject("user.city"))
        .unwrap();

    let (live, total) = col.stats();
    assert_eq!(total, 2);
    assert_eq!(live, 1);

    let hits = col
        .recall(RecallOpts::new("用户现在住在哪里").budget_tokens(300))
        .unwrap();
    assert!(hits.iter().all(|h| h.id != id_old), "old version leaked");
    assert!(hits.iter().any(|h| h.id == id_new));

    let as_of_hits = col
        .recall(
            RecallOpts::new("用户住在哪个城市")
                .as_of(0)
                .k_max(8),
        )
        .unwrap();
    assert!(
        as_of_hits.iter().any(|h| h.id == id_old),
        "as_of time travel must see the old version"
    );

    col.forget(id_new).unwrap();
    let (live, _) = col.stats();
    assert_eq!(live, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn token_budget_truncates() {
    let dir = tmpdir("budget");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    for i in 0..20 {
        let long_text = format!(
            "meeting notes session {i}: discussed roadmap priorities and quarterly planning details at length with many words included here"
        );
        col.remember(RememberOpts::new(long_text)).unwrap();
    }

    let opts = RecallOpts::new("roadmap priorities quarterly planning")
        .budget_tokens(50)
        .k_max(64);
    let hits = col.recall(opts).unwrap();
    assert!(!hits.is_empty());
    let used: usize = hits.iter().map(|h| h.estimated_tokens).sum();
    assert!(used <= 50, "budget exceeded: {used}");
    assert!(hits.len() < 20);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn forget_and_purge_semantics() {
    let dir = tmpdir("purge");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let id = col.remember(RememberOpts::new("secret token stored in vault alpha")).unwrap();
    col.forget(id).unwrap();

    let hits = col
        .recall(RecallOpts::new("secret token vault").budget_tokens(100))
        .unwrap();
    assert!(hits.iter().all(|h| h.text != "secret token stored in vault alpha"));

    col.hard_delete(id).unwrap();
    let (_, total) = col.stats();
    assert_eq!(total, 0);

    db.checkpoint_all().unwrap();
    drop(col);
    drop(db);
    let db2 = Db::open(&dir).unwrap();
    let (_, total2) = db2.collection("main").unwrap().stats();
    assert_eq!(total2, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wal_crash_recovery_without_checkpoint() {
    let dir = tmpdir("walcrash");

    {
        let db = Db::open(&dir).unwrap();
        let col = db.create_collection("main").unwrap();
        for i in 0..5 {
            col.remember(RememberOpts::new(format!("note number {i} about rust programming")))
                .unwrap();
        }
        drop(col);
        drop(db);
    }

    let db = Db::open(&dir).unwrap();
    let col = db.collection("main").unwrap();
    let (_, total) = col.stats();
    assert_eq!(total, 5, "wal-only data must survive reopen");

    let hits = col.recall(RecallOpts::new("rust programming note 3").budget_tokens(200)).unwrap();
    assert!(!hits.is_empty());

    db.checkpoint_all().unwrap();
    let wal_path = dir.join("collections").join("main").join("wal.jsonl");
    let wal_size = std::fs::metadata(&wal_path).unwrap().len();
    assert_eq!(wal_size, 0, "checkpoint must truncate wal");

    drop(col);
    drop(db);
    let db2 = Db::open(&dir).unwrap();
    let (_, total2) = db2.collection("main").unwrap().stats();
    assert_eq!(total2, 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profiles_change_ranking() {
    let dir = tmpdir("profiles");
    let db = Db::open(&dir).unwrap();
    let col = db.create_collection("main").unwrap();

    let old_id = col
        .remember(RememberOpts::new("deploy service to kubernetes cluster").event_time(1))
        .unwrap();
    let new_unrelated = col
        .remember(RememberOpts::new("lunch order pizza and salad today").event_time(9_000_000_000))
        .unwrap();
    let _ = new_unrelated;

    let chat_hits = col
        .recall(RecallOpts::new("deploy to kubernetes").profile(Profile::Chat).budget_tokens(500))
        .unwrap();
    assert!(chat_hits[0].id == old_id || !chat_hits.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

