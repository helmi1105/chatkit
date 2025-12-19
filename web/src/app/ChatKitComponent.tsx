"use client"

import { useState, useCallback, useEffect } from "react"
import { ChatKit, useChatKit, type ColorScheme } from "@openai/chatkit-react"

type ChatKitComponentProps = {
  userId: string
  theme: ColorScheme
  maximize: boolean
  onResponseEnd: () => void
}

type MapState = {
  lat: number
  lng: number
  zoom: number
} | null

function ChatKitComponent({
  userId,
  theme,
  maximize,
  onResponseEnd,
}: ChatKitComponentProps) {
  const [error, setError] = useState<string | null>(null)
  const [mapState, setMapState] = useState<MapState>(null)
  const [plotHtml, setPlotHtml] = useState<string | null>(null) // 👈 NEW: HTML for Plotly chart

  // --- script availability effect (unchanged) ---
  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }

    let timeoutId: number | undefined

    const handleLoaded = () => {
      console.log("script loaded")
      setError(null)
    }

    const handleError = () => {
      setError("Failed to load chatkit.js.")
    }

    window.addEventListener("chatkit-script-loaded", handleLoaded)
    window.addEventListener("chatkit-script-error", handleError)

    if (window.customElements?.get("openai-chatkit")) {
      handleLoaded()
    } else if (error === null) {
      timeoutId = window.setTimeout(() => {
        if (!window.customElements?.get("openai-chatkit")) {
          handleError()
        }
      }, 5000)
    }

    return () => {
      window.removeEventListener("chatkit-script-loaded", handleLoaded)
      window.removeEventListener("chatkit-script-error", handleError)
      if (timeoutId) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [typeof window])

  function buildErrorResponse(message: string): Response {
    console.error(message)
    return new Response(
      JSON.stringify({
        error: message,
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    )
  }


  function makePlotlyResponsive(rawHtml: string): string {
  if (!rawHtml) return rawHtml

  let html = rawHtml

  // 1) Rewrite the inline style on the Plotly div
  //    from: style="height:600px; width:800px;"
  //    to:   style="height:100%; width:100%; max-width:100%; max-height:100%;"
  html = html.replace(
    /(<div[^>]*class="plotly-graph-div"[^>]*style=")([^"]*)(")/,
    (_match, start, _style, end) =>
      `${start}height:100%; width:100%; max-width:100%; max-height:100%;${end}`,
  )

  // 2) Remove explicit width/height from layout JSON (…,"width":800,"height":600,…)
  html = html.replace(/"width"\s*:\s*\d+\s*,\s*"height"\s*:\s*\d+\s*,?/, "")

  // 3) Ensure html/body fill the iframe
  const styleBlock = `
<style>
  html, body {
    margin: 0;
    padding: 0;
    height: 100%;
  }
</style>
`

  if (html.includes("</head>")) {
    html = html.replace("</head>", `${styleBlock}</head>`)
  } else {
    html = `<!DOCTYPE html><html><head>${styleBlock}</head><body>${html}</body></html>`
  }

  return html
}



  const _fetch = useCallback(
    async function customFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
      let requestInit: RequestInit = init ?? {}
      requestInit.headers = {
        userId: userId,
      }
      try {
        const response = await fetch(input, requestInit)

        if (!response.ok) {
          const jsonResponse = await response.json()
          throw new Error(`${response.statusText}: ${JSON.stringify(jsonResponse)}`)
        }

        return response
      } catch (error) {
        console.log("error: ", error)
        const errorMessage =
          error instanceof Error ? error.message : "Unable to start fetch from the server."

        setError(errorMessage)
        return buildErrorResponse(errorMessage)
      }
    },
    [userId],
  )

  const chatkit = useChatKit({
    api: {
      url: "http://127.0.0.1:8000/chatkit",
      domainKey: "localhost",
      fetch: _fetch,
    },
    theme: {
      colorScheme: theme,
    },
    startScreen: {
      greeting: "Hey, what can I do for you?",
      prompts: [
        {
          label: "Hello!",
          prompt: "Hello!",
          icon: "star-filled",
        },
        {
          label: "What can you do?",
          prompt: "What can you do?",
          icon: "circle-question",
        },
      ],
    },
    composer: {
      placeholder: "Your Agent is ready!",
      attachments: {
        enabled: false,
      },
    },
    threadItemActions: {
      feedback: true,
      retry: true,
    },

    onClientTool: async (invocation) => {
      if (invocation.name === "your_client_function_name") {
        return {}
      }
      return {}
    },
    

    // 🔹 Handle widget actions (map + plotly)
    widgets: {
      onAction: async (action, widgetItem) => {
        console.log("Widget action:", action, widgetItem)

        if (action.type === "map.show_inline") {
          const lat = Number(action.payload?.lat ?? 51.5)
          const lng = Number(action.payload?.lng ?? -0.09)
          const zoom = Number(action.payload?.zoom ?? 13)
          setMapState({ lat, lng, zoom })
          return
        }

        if (action.type === "report.open") {
          const html = String(action.payload?.html ?? "")
          if (html) {
            const responsiveHtml = makePlotlyResponsive(html)
            setPlotHtml(responsiveHtml)
          }
          return
        }
      },
    },

    onResponseEnd: () => {
      onResponseEnd()
    },
    onResponseStart: () => {
      setError(null)
    },
    onThreadLoadStart: (event) => {
      console.log("Thread load started: ", event.threadId)
    },
    onThreadLoadEnd: (event) => {
      console.log("Thread load ended: ", event.threadId)
    },
    onThreadChange: (event) => {
      console.log("Thread changed: ", event.threadId)
    },
    onError: (event) => {
      console.error("ChatKit error: ", event.error)
    },
    onLog: (event) => {
      console.log(`[${event.name}]: `, event.data)
    },
  })

  // Build the OSM URL from state
  const mapUrl =
    mapState !== null
      ? (() => {
          const { lat, lng, zoom } = mapState
          const delta = 0.02
          const bbox = [lng - delta, lat - delta, lng + delta, lat + delta].join(",")
          return `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lat},${lng}`
        })()
      : null

  const hasMap = !!mapUrl
  const hasPlot = !!plotHtml
  const showRightPane = hasMap || hasPlot

return (
  <div
    className={`relative h-full rounded-2xl overflow-hidden bg-white shadow-sm dark:bg-slate-900 ${
      maximize ? "w-full" : "w-80 ml-auto"
    }`}
  >
    <div className="flex h-full w-full">
      <div className={showRightPane ? "w-1/2 border-r border-slate-200" : "w-full"}>
        <ChatKit control={chatkit.control} className="block h-full w-full" />
      </div>

      {showRightPane && (
        <div className="w-1/2 h-full flex flex-col bg-slate-100 dark:bg-slate-950">
          {/* Header */}
          <div className="shrink-0 flex items-center justify-between px-2 py-1 text-xs text-slate-600 dark:text-slate-300">
            <span>
              {hasPlot && "Plotly chart"}
              {!hasPlot && hasMap && mapState && (
                <>
                  Map: {mapState.lat.toFixed(4)}, {mapState.lng.toFixed(4)} (z={mapState.zoom})
                </>
              )}
            </span>
            <button
              type="button"
              className="
                inline-flex items-center gap-1.5
                rounded-full
                bg-slate-900/90 dark:bg-slate-800/90
                px-3 py-1
                text-[11px] font-medium
                text-slate-50 dark:text-slate-100
                shadow-sm
                ring-1 ring-slate-700/70 dark:ring-slate-600/70
                hover:bg-slate-900 hover:dark:bg-slate-700
                hover:ring-slate-500/80
                transition-colors transition-shadow
              "
              onClick={() => {
                setMapState(null)
                setPlotHtml(null)
              }}
            >
              <span className="text-[13px] leading-none">×</span>
              <span>Close</span>
            </button>
          </div>

          {/* Content: fills remaining height */}
          <div className="flex-1 min-h-0">
            {hasPlot ? (
              <iframe
                title="Plotly chart"
                srcDoc={plotHtml ?? ""}
                className="h-full w-full border-0 bg-white"
                sandbox="allow-scripts allow-same-origin"
              />
            ) : hasMap ? (
              <iframe
                title="Inline OSM map"
                src={mapUrl as string}
                className="h-full w-full border-0"
                loading="lazy"
              />
            ) : null}
          </div>
        </div>
      )}
    </div>

    {/* error overlay ... (unchanged) */}
  </div>
)

}

export default ChatKitComponent
