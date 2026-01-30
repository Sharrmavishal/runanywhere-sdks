package com.runanywhere.runanywhereai.http

import android.util.Log
import com.runanywhere.sdk.public.RunAnywhere
import com.runanywhere.sdk.public.extensions.LLM.LLMGenerationOptions
import com.runanywhere.sdk.public.extensions.generate
import com.runanywhere.sdk.public.extensions.availableModels
import com.runanywhere.sdk.public.extensions.Models.ModelInfo as SDKModelInfo
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * HTTP Server wrapper for RunAnywhere SDK inference.
 * Exposes AI inference via REST API for remote access testing.
 *
 * Endpoints:
 * - POST /inference - Text generation
 * - GET /health - Service status
 * - GET /models - List available models
 */
class InferenceHttpServer(
    private val port: Int = 8080,
) {
    private var server: EmbeddedServer<*, *>? = null
    private val serverScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val tag = "InferenceHttpServer"

    /**
     * Start the HTTP server
     */
    fun start() {
        if (server != null) {
            Log.w(tag, "⚠️ Server already running")
            return
        }

        serverScope.launch {
            try {
                server = embeddedServer(Netty, port = port, host = "0.0.0.0") {
                    install(ContentNegotiation) {
                        json(
                            Json {
                                prettyPrint = true
                                isLenient = true
                                ignoreUnknownKeys = true
                            },
                        )
                    }

                    routing {
                        // Health check endpoint
                        get("/health") {
                            val isSDKReady = RunAnywhere.isInitialized
                            val response =
                                HealthResponse(
                                    status = if (isSDKReady) "ready" else "not_ready",
                                    sdkInitialized = isSDKReady,
                                )
                            call.respond(HttpStatusCode.OK, response)
                        }

                        // List available models
                        get("/models") {
                            try {
                                val models: List<SDKModelInfo> = RunAnywhere.availableModels()
                                val response = ModelsResponse(models = models.map { model: SDKModelInfo ->
                                    ModelInfo(
                                        id = model.id,
                                        name = model.name,
                                        framework = model.framework.rawValue,
                                        isDownloaded = model.isDownloaded,
                                    )
                                })
                                call.respond(HttpStatusCode.OK, response)
                            } catch (e: Exception) {
                                Log.e(tag, "Error listing models: ${e.message}", e)
                                call.respond(
                                    HttpStatusCode.InternalServerError,
                                    ErrorResponse(error = "Failed to list models: ${e.message}"),
                                )
                            }
                        }

                        // Inference endpoint
                        post("/inference") {
                            try {
                                val request = call.receive<InferenceRequest>()

                                if (!RunAnywhere.isInitialized) {
                                    call.respond(
                                        HttpStatusCode.ServiceUnavailable,
                                        ErrorResponse(error = "SDK not initialized"),
                                    )
                                    return@post
                                }

                                Log.i(tag, "📥 Inference request: prompt='${request.prompt.take(50)}...', maxTokens=${request.maxTokens}")

                                // Perform inference
                                val options =
                                    LLMGenerationOptions(
                                        maxTokens = request.maxTokens ?: 100,
                                        temperature = request.temperature ?: 0.7f,
                                    )

                                val result = RunAnywhere.generate(request.prompt, options)

                                val response =
                                    InferenceResponse(
                                        text = result.text,
                                        tokensUsed = result.tokensUsed,
                                        latencyMs = result.latencyMs,
                                        modelUsed = result.modelUsed,
                                        tokensPerSecond = result.tokensPerSecond,
                                    )

                                Log.i(tag, "✅ Inference complete: ${result.tokensUsed} tokens in ${result.latencyMs}ms")
                                call.respond(HttpStatusCode.OK, response)
                            } catch (e: Exception) {
                                Log.e(tag, "❌ Inference error: ${e.message}", e)
                                call.respond(
                                    HttpStatusCode.InternalServerError,
                                    ErrorResponse(error = "Inference failed: ${e.message}"),
                                )
                            }
                        }
                    }
                }.start(wait = false)

                Log.i(tag, "🚀 HTTP server started on port $port")
                Log.i(tag, "   Health: http://<device-ip>:$port/health")
                Log.i(tag, "   Models: http://<device-ip>:$port/models")
                Log.i(tag, "   Inference: http://<device-ip>:$port/inference")
            } catch (e: Exception) {
                Log.e(tag, "❌ Failed to start HTTP server: ${e.message}", e)
            }
        }
    }

    /**
     * Stop the HTTP server
     */
    fun stop() {
        serverScope.launch {
            try {
                server?.stop(1000, 2000)
                server = null
                Log.i(tag, "🛑 HTTP server stopped")
            } catch (e: Exception) {
                Log.e(tag, "Error stopping server: ${e.message}", e)
            }
        }
    }

    /**
     * Get the server's local IP address (for display purposes)
     */
    fun getServerUrl(): String {
        return "http://<device-ip>:$port"
    }

    /**
     * Cleanup resources (called when object is garbage collected)
     * Note: In Android, prefer explicit cleanup in onTerminate()
     */
    protected fun finalize() {
        try {
            stop()
            serverScope.cancel()
        } catch (e: Exception) {
            // Ignore errors during finalization
        }
    }
}

// Request/Response DTOs

@Serializable
data class InferenceRequest(
    val prompt: String,
    val maxTokens: Int? = null,
    val temperature: Float? = null,
)

@Serializable
data class InferenceResponse(
    val text: String,
    val tokensUsed: Int,
    val latencyMs: Double,
    val modelUsed: String,
    val tokensPerSecond: Double,
)

@Serializable
data class HealthResponse(
    val status: String,
    val sdkInitialized: Boolean,
)

@Serializable
data class ModelsResponse(
    val models: List<ModelInfo>,
)

@Serializable
data class ModelInfo(
    val id: String,
    val name: String,
    val framework: String,
    val isDownloaded: Boolean,
)

@Serializable
data class ErrorResponse(
    val error: String,
)
