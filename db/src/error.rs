#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("collection not found: {0}")]
    CollectionNotFound(String),
    #[error("id not found: {0}")]
    NotFound(u64),
    #[error("locked: {0}")]
    Locked(String),
}

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

