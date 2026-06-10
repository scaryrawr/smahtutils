use rmcp::{
    Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};

use crate::{
    model::{IndexResponse, IndexedListResponse, QueryResponse, ServiceStatus},
    service::{self, AppState, IndexRequest, ListIndexedRequest, QueryRequest},
};

#[derive(Clone)]
pub struct SmahtiesMcp {
    state: AppState,
    tool_router: ToolRouter<Self>,
}

impl SmahtiesMcp {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(router = tool_router)]
impl SmahtiesMcp {
    #[tool(
        description = "Search indexed code with semantic, keyword, or hybrid ranking. Returns ranked code units with source snippets."
    )]
    async fn query_code(
        &self,
        Parameters(request): Parameters<QueryRequest>,
    ) -> std::result::Result<Json<QueryResponse>, String> {
        service::query_code(&self.state, request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        description = "Request high-priority indexing for a file or directory path under the active smahties runtime scope."
    )]
    async fn index_path(
        &self,
        Parameters(request): Parameters<IndexRequest>,
    ) -> std::result::Result<Json<IndexResponse>, String> {
        service::index_path(&self.state, request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        description = "Show indexing status, queue counts, recent errors, and active indexer lease state."
    )]
    async fn status(&self) -> std::result::Result<Json<ServiceStatus>, String> {
        service::status(&self.state)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        description = "List indexed code units with path/language filters and bounded pagination. Source is omitted unless include_source is true and limit <= 20."
    )]
    fn list_indexed(
        &self,
        Parameters(request): Parameters<ListIndexedRequest>,
    ) -> std::result::Result<Json<IndexedListResponse>, String> {
        service::list_indexed(&self.state, request)
            .map(Json)
            .map_err(|error| error.to_string())
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SmahtiesMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Local semantic code search scoped to the directory where smahties is running.",
        )
    }
}

pub async fn serve(state: AppState) -> crate::Result<()> {
    let service = SmahtiesMcp::new(state)
        .serve(stdio())
        .await
        .map_err(|error| {
            crate::SmahtiesError::InvalidRequest(format!("failed to start MCP service: {error}"))
        })?;
    service.waiting().await.map_err(|error| {
        crate::SmahtiesError::InvalidRequest(format!("MCP service failed: {error}"))
    })?;
    Ok(())
}
