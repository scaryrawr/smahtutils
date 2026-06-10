use axum::{
    Json, Router,
    extract::{Query as QueryParams, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;

use crate::{
    SmahtiesError,
    model::{IndexResponse, IndexedListResponse, QueryResponse, ServiceStatus},
    service::{self, AppState, IndexRequest, ListIndexedRequest, QueryRequest},
};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/index", post(index))
        .route("/query", get(query_get).post(query_post))
        .route("/indexed", get(list_indexed))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}

async fn status(State(state): State<AppState>) -> Result<Json<ServiceStatus>, ApiError> {
    Ok(Json(service::status(&state).await?))
}

async fn index(
    State(state): State<AppState>,
    Json(request): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, ApiError> {
    Ok(Json(service::index_path(&state, request).await?))
}

async fn query_get(
    State(state): State<AppState>,
    QueryParams(request): QueryParams<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    Ok(Json(service::query_code(&state, request).await?))
}

async fn query_post(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    Ok(Json(service::query_code(&state, request).await?))
}

async fn list_indexed(
    State(state): State<AppState>,
    QueryParams(request): QueryParams<ListIndexedRequest>,
) -> Result<Json<IndexedListResponse>, ApiError> {
    Ok(Json(service::list_indexed(&state, request)?))
}

#[derive(Debug)]
pub struct ApiError(SmahtiesError);

impl From<SmahtiesError> for ApiError {
    fn from(value: SmahtiesError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            SmahtiesError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (
            status,
            Json(ErrorResponse {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
}
