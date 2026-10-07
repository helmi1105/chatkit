// Keep the deadline active until the response body ends, including SSE streams.
export async function timedFetch(input: string | URL | Request, init: RequestInit, timeoutMs = 210_000): Promise<Response> {
  const controller = new AbortController()
  const upstream = init.signal ?? (input instanceof Request ? input.signal : undefined)
  const cancel = () => controller.abort(upstream?.reason)
  upstream?.addEventListener('abort', cancel, { once: true })
  if (upstream?.aborted) cancel()
  const timer = setTimeout(() => controller.abort(new Error('La réponse a pris trop de temps. Veuillez réessayer.')), timeoutMs)
  const cleanup = () => {
    clearTimeout(timer)
    upstream?.removeEventListener('abort', cancel)
  }
  try {
    const response = await fetch(input, { ...init, signal: controller.signal })
    if (!response.body) { cleanup(); return response }
    const reader = response.body.getReader()
    const body = new ReadableStream<Uint8Array>({
      async pull(stream) {
        try {
          const result = await reader.read()
          if (result.done) { cleanup(); stream.close() }
          else stream.enqueue(result.value)
        } catch (error) {
          cleanup()
          stream.error(controller.signal.aborted ? controller.signal.reason : error)
        }
      },
      async cancel(reason) {
        cleanup()
        controller.abort(reason)
        await reader.cancel(reason)
      },
    })
    return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers })
  } catch (error) {
    cleanup()
    throw controller.signal.aborted ? controller.signal.reason : error
  }
}
