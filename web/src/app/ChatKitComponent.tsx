"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { ChatKit, useChatKit, type ColorScheme } from "@openai/chatkit-react"
import GuidedTour, { TOUR_STEPS } from "./GuidedTour"
import { timedFetch } from "./timedFetch"

type ChatKitComponentProps = {
  userId: string
  theme: ColorScheme
}

type MapState = {
  lat: number
  lng: number
  zoom: number
} | null

type Progress = {
  course_title: string
  current_kc: { id: string; title: string; index: number; total: number } | null
  module_title: string | null
  validated: number
  total: number
  diagnostic_done: boolean
  quiz_pending: boolean
}

const PROVIDER_STORAGE_KEY = "chatkit-provider"
const API_KEY_STORAGE_KEY = "chatkit-provider-api-key"
const TOUR_SEEN_KEY = "chatkit-tour-seen"
const ACCESS_CODE_KEY = "chatkit-access-code"
const PROGRESS_POLL_MS = 30_000

const API_BASE = "/backend"

function readLocal(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeLocal(key: string, value: string | null): void {
  try {
    if (value === null) {
      window.localStorage.removeItem(key)
    } else {
      window.localStorage.setItem(key, value)
    }
  } catch {
    // storage unavailable (private browsing, quota): the value only lasts this render
  }
}

function makePlotlyResponsive(rawHtml: string): string {
  if (!rawHtml) return rawHtml

  let html = rawHtml

  // 1) Rewrite the inline style on the Plotly div to fill the iframe
  html = html.replace(
    /(<div[^>]*class="plotly-graph-div"[^>]*style=")([^"]*)(")/,
    (_match, start, _style, end) =>
      `${start}height:100%; width:100%; max-width:100%; max-height:100%;${end}`,
  )

  // 2) Remove explicit width/height from layout JSON
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

function progressLabel(p: Progress | null): string {
  if (!p) return ""
  const parts: string[] = []
  if (!p.diagnostic_done && !p.current_kc) {
    parts.push("Diagnostic non fait")
  } else if (p.current_kc) {
    parts.push(`Notion ${p.current_kc.index}/${p.current_kc.total} : ${p.current_kc.title}`)
  }
  if (p.module_title) {
    parts.push(`Chapitre : ${p.module_title}`)
  }
  const n = p.validated
  parts.push(`${n} notion${n > 1 ? "s" : ""} validée${n > 1 ? "s" : ""} sur ${p.total}`)
  if (p.quiz_pending) {
    parts.push("Quiz en attente de vos réponses")
  }
  return parts.join(" · ")
}

function ChatKitComponent({ userId, theme }: ChatKitComponentProps) {
  const [scriptError, setScriptError] = useState<string | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [mapState, setMapState] = useState<MapState>(null)
  const [plotHtml, setPlotHtml] = useState<string | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [showTour, setShowTour] = useState(false)
  const [tourStep, setTourStep] = useState(0)
  const [showHelpMenu, setShowHelpMenu] = useState(false)
  const [showSettingsMenu, setShowSettingsMenu] = useState(false)
  const [provider, setProvider] = useState<string>("mistral")
  const [apiKey, setApiKey] = useState<string>("")
  const [progress, setProgress] = useState<Progress | null>(null)

  // Access-code gate: null = unknown (health not fetched yet), false = not
  // required, true = required. accessCode is what the user typed (persisted).
  const [accessRequired, setAccessRequired] = useState<boolean | null>(null)
  const [accessCode, setAccessCode] = useState<string>("")
  const [accessMessage, setAccessMessage] = useState<string | null>(null)
  const [codeInput, setCodeInput] = useState<string>("")
  const accessCodeRef = useRef<string>("")
  accessCodeRef.current = accessCode

  const docsUrl = `${API_BASE}/static/guide_fr.html`

  // --- first-visit state (tour, provider, key, access code) ---
  useEffect(() => {
    const storedProvider = readLocal(PROVIDER_STORAGE_KEY)
    if (storedProvider) setProvider(storedProvider)
    try {
      // The API key stays in sessionStorage on purpose: a third-party secret
      // must not outlive the tab.
      const storedKey = window.sessionStorage.getItem(API_KEY_STORAGE_KEY)
      if (storedKey) setApiKey(storedKey)
    } catch {
      // ignore
    }
    const storedCode = readLocal(ACCESS_CODE_KEY)
    if (storedCode) setAccessCode(storedCode)
    if (!readLocal(TOUR_SEEN_KEY)) {
      setShowTour(true)
    }
  }, [])

  // --- health check: is an access code required? ---
  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data: { access_code_required?: boolean }) => {
        if (!cancelled) setAccessRequired(Boolean(data.access_code_required))
      })
      .catch(() => {
        // Backend unreachable or old backend without /health: do not block the chat.
        if (!cancelled) setAccessRequired(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const updateProvider = useCallback((next: string) => {
    setProvider(next)
    writeLocal(PROVIDER_STORAGE_KEY, next)
  }, [])

  const updateApiKey = useCallback((next: string) => {
    setApiKey(next)
    try {
      if (next) {
        window.sessionStorage.setItem(API_KEY_STORAGE_KEY, next)
      } else {
        window.sessionStorage.removeItem(API_KEY_STORAGE_KEY)
      }
    } catch {
      // ignore
    }
  }, [])

  const closeTour = useCallback(() => {
    setShowTour(false)
    writeLocal(TOUR_SEEN_KEY, "1")
  }, [])

  const openTour = useCallback(() => {
    setTourStep(0)
    setShowTour(true)
  }, [])

  const dropAccessCode = useCallback((message: string) => {
    setAccessCode("")
    writeLocal(ACCESS_CODE_KEY, null)
    setAccessRequired(true)
    setAccessMessage(message)
  }, [])

  // --- chatkit.js availability (the custom element must be defined) ---
  useEffect(() => {
    let timeoutId: number | undefined
    let cancelled = false
    if (window.customElements?.get("openai-chatkit")) {
      setScriptError(null)
    } else {
      window.customElements
        ?.whenDefined("openai-chatkit")
        .then(() => {
          if (!cancelled) setScriptError(null)
        })
        .catch(() => undefined)
      timeoutId = window.setTimeout(() => {
        if (!cancelled && !window.customElements?.get("openai-chatkit")) {
          setScriptError("Le composant de chat (chatkit.js) n'a pas pu être chargé. Vérifiez votre connexion puis réessayez.")
        }
      }, 8000)
    }
    return () => {
      cancelled = true
      if (timeoutId) window.clearTimeout(timeoutId)
    }
  }, [])

  const buildHeaders = useCallback((): Record<string, string> => {
    const headers: Record<string, string> = {
      "X-ChatKit-Request": "1",
      "X-Provider": provider,
    }
    if (apiKey) headers["X-Provider-Api-Key"] = apiKey
    if (accessCodeRef.current) headers["X-Access-Code"] = accessCodeRef.current
    return headers
  }, [userId, provider, apiKey])

  // --- progress bar ---
  const refreshProgress = useCallback(async () => {
    if (accessRequired === null) return
    if (accessRequired && !accessCodeRef.current) return
    try {
      const response = await fetch(`${API_BASE}/progress`, { headers: buildHeaders() })
      if (response.status === 401) {
        const body = await response.clone().json().catch(() => ({}))
        if (body.code === 'login_required') {
          window.dispatchEvent(new Event('chatkit-session-expired'))
          return
        }
        dropAccessCode("Code d'accès refusé ou expiré.")
        return
      }
      if (!response.ok) return
      const data = (await response.json()) as Progress
      setProgress(data)
    } catch {
      // progress is a convenience; a failed refresh is not an error for the learner
    }
  }, [accessRequired, buildHeaders, dropAccessCode])

  useEffect(() => {
    refreshProgress()
    const tick = () => {
      if (document.visibilityState === "visible") refreshProgress()
    }
    const intervalId = window.setInterval(tick, PROGRESS_POLL_MS)
    document.addEventListener("visibilitychange", tick)
    return () => {
      window.clearInterval(intervalId)
      document.removeEventListener("visibilitychange", tick)
    }
  }, [refreshProgress])

  function buildErrorResponse(message: string): Response {
    console.error(message)
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }

  const _fetch = useCallback(
    async function customFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
      const requestInit: RequestInit = { ...(init ?? {}) }
      const headers = new Headers(input instanceof Request ? input.headers : undefined)
      new Headers(init?.headers).forEach((value, key) => headers.set(key, value))
      Object.entries(buildHeaders()).forEach(([key, value]) => headers.set(key, value))
      requestInit.headers = headers
      requestInit.credentials = 'same-origin'
      try {
        const response = await timedFetch(input, requestInit)

        if (!response.ok) {
          let message = `Erreur ${response.status}`
          try {
            const body = await response.json()
            if (body?.code === 'login_required') window.dispatchEvent(new Event('chatkit-session-expired'))
            message = String(body?.message ?? body?.error ?? message)
          } catch {
            // non-JSON body: keep the status message
          }
          if (response.status === 401) {
            dropAccessCode(message)
          }
          throw new Error(message)
        }

        return response
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Impossible de joindre le serveur."
        setRequestError(errorMessage)
        return buildErrorResponse(errorMessage)
      }
    },
    [buildHeaders, dropAccessCode],
  )

  const chatkit = useChatKit({
    api: {
      // Keep API requests on the frontend origin so HttpOnly cookies are first-party.
      url: `${typeof window === 'undefined' ? 'http://localhost:3000' : window.location.origin}${API_BASE}/chatkit`,
      domainKey: process.env.NEXT_PUBLIC_CHATKIT_DOMAIN_KEY ?? "localhost",
      fetch: _fetch,
      uploadStrategy: { type: "two_phase" },
    },
    theme: {
      colorScheme: theme,
    },
    startScreen: {
      greeting: "Bonjour. Choisissez une suggestion pour commencer, ou écrivez directement votre question.",
      prompts: [
        {
          label: "Commencer le diagnostic",
          prompt: "commencer le diagnostic",
          icon: "star-filled",
        },
        {
          label: "Où en suis-je ?",
          prompt: "ma progression",
          icon: "chart",
        },
        {
          label: "Poser une question sur un symbole",
          prompt: "Que signifie un triangle ?",
          icon: "circle-question",
        },
        {
          label: "Aide",
          prompt: "aide",
          icon: "book-open",
        },
      ],
    },
    composer: {
      placeholder: "Écrivez votre message ou votre question, ou joignez une photo de symbole",
      attachments: {
        enabled: true,
      },
    },
    threadItemActions: {
      feedback: true,
      retry: true,
    },

    widgets: {
      onAction: async (action) => {
        if (action.type === "map.show_inline") {
          const lat = Number(action.payload?.lat ?? 51.5)
          const lng = Number(action.payload?.lng ?? -0.09)
          const zoom = Number(action.payload?.zoom ?? 13)
          setMapState({ lat, lng, zoom })
          return
        }

        if (action.type === "report.open") {
          const url = String(action.payload?.url ?? "")
          if (url) {
            setPdfUrl(url)
            setPlotHtml(null)
            setMapState(null)
            return
          }
          const html = String(action.payload?.html ?? "")
          if (html) {
            setPlotHtml(makePlotlyResponsive(html))
            setPdfUrl(null)
          }
          return
        }

        if (action.type === "radar.click") {
          const html = String(action.payload?.html ?? "")
          if (html) {
            setPlotHtml(makePlotlyResponsive(html))
          }
          return
        }
      },
    },

    onResponseEnd: () => {
      refreshProgress()
    },
    onResponseStart: () => {
      setRequestError(null)
      if (showTour) closeTour()
    },
    onError: (event) => {
      console.error("ChatKit error: ", event.error)
      setRequestError(event.error.message || 'La réponse a été interrompue. Veuillez réessayer.')
    },
  })

  const sendCommand = useCallback(
    (text: string) => {
      chatkit.sendUserMessage({ text })
    },
    [chatkit],
  )

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
  const hasPdf = !!pdfUrl
  const showRightPane = hasMap || hasPlot || hasPdf

  const gateOpen = accessRequired === true && !accessCode
  const statusText = progressLabel(progress)

  const submitAccessCode = (event: React.FormEvent) => {
    event.preventDefault()
    const code = codeInput.trim()
    if (!code) return
    setAccessCode(code)
    writeLocal(ACCESS_CODE_KEY, code)
    setAccessMessage(null)
    setCodeInput("")
  }

  return (
    <div className="relative flex h-full min-h-0 w-full flex-1 flex-col overflow-hidden rounded-2xl bg-white shadow-sm dark:bg-slate-900">
      {/* Top bar: progress on the left, tools on the right */}
      <div className="flex shrink-0 items-center gap-2 border-b border-slate-200 px-3 py-2 dark:border-slate-700">
        <div
          className="min-w-0 flex-1 truncate text-[11px] text-slate-600 dark:text-slate-300"
          title={statusText}
        >
          {statusText}
        </div>
        <button
          type="button"
          onClick={openTour}
          className="hidden shrink-0 rounded-full bg-slate-900/90 px-3 py-1 text-[11px] font-medium text-slate-50 shadow-sm ring-1 ring-slate-700/70 hover:bg-slate-900 sm:inline-flex dark:bg-slate-800/90 dark:text-slate-100 dark:ring-slate-600/70"
        >
          Visite guidée
        </button>
        <div className="relative shrink-0">
          <button
            type="button"
            aria-label="Réglages du modèle"
            aria-expanded={showSettingsMenu}
            onClick={() => {
              setShowSettingsMenu((v) => !v)
              setShowHelpMenu(false)
            }}
            className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-900/90 text-[11px] font-semibold text-slate-50 shadow-sm ring-1 ring-slate-700/70 hover:bg-slate-900 dark:bg-slate-800/90 dark:text-slate-100 dark:ring-slate-600/70"
          >
            ⚙
          </button>
          {showSettingsMenu && (
            <div className="absolute right-0 z-30 mt-1.5 w-64 space-y-2 rounded-xl border border-slate-200 bg-white p-3 text-xs shadow-lg dark:border-slate-700 dark:bg-slate-800">
              <div>
                <label className="mb-1 block font-medium text-slate-700 dark:text-slate-200">
                  Modèle
                </label>
                <select
                  value={provider}
                  onChange={(e) => updateProvider(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1 text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                >
                  <option value="mistral">Mistral (par défaut)</option>
                  <option value="openai">GPT-4.1 (OpenAI)</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block font-medium text-slate-700 dark:text-slate-200">
                  Ma clé API (si plus de quota partagé)
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => updateApiKey(e.target.value)}
                  placeholder="sk-... ou clé Mistral"
                  className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1 text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                />
                <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-400">
                  Conservée dans cet onglet uniquement, envoyée à ce serveur seul.
                </p>
              </div>
              <div className="border-t border-slate-200 pt-2 dark:border-slate-700">
                <button
                  type="button"
                  onClick={() => {
                    setShowSettingsMenu(false)
                    sendCommand("recommencer à zéro")
                  }}
                  className="block w-full rounded-lg px-2.5 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700"
                >
                  Recommencer à zéro
                </button>
                <p className="mt-1 px-2.5 text-[10px] text-slate-500 dark:text-slate-400">
                  Efface votre progression après confirmation.
                </p>
              </div>
            </div>
          )}
        </div>
        <div className="relative shrink-0">
          <button
            type="button"
            aria-label="Aide"
            aria-expanded={showHelpMenu}
            onClick={() => {
              setShowHelpMenu((v) => !v)
              setShowSettingsMenu(false)
            }}
            className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-900/90 text-[11px] font-semibold text-slate-50 shadow-sm ring-1 ring-slate-700/70 hover:bg-slate-900 dark:bg-slate-800/90 dark:text-slate-100 dark:ring-slate-600/70"
          >
            ?
          </button>
          {showHelpMenu && (
            <div className="absolute right-0 z-30 mt-1.5 w-52 rounded-xl border border-slate-200 bg-white p-1.5 text-xs shadow-lg dark:border-slate-700 dark:bg-slate-800">
              <button
                type="button"
                onClick={() => {
                  setShowHelpMenu(false)
                  openTour()
                }}
                className="block w-full rounded-lg px-2.5 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                Revoir la visite guidée
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowHelpMenu(false)
                  sendCommand("aide")
                }}
                className="block w-full rounded-lg px-2.5 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                Aide dans le chat
              </button>
              <a
                href={docsUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setShowHelpMenu(false)}
                className="block w-full rounded-lg px-2.5 py-1.5 text-left text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700"
              >
                Documentation complète
              </a>
            </div>
          )}
        </div>
      </div>

      {requestError && (
        <div
          role="alert"
          className="flex shrink-0 items-start gap-2 border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
        >
          <span className="min-w-0 flex-1 break-words">{requestError}</span>
          <button
            type="button"
            aria-label="Fermer"
            onClick={() => setRequestError(null)}
            className="shrink-0 rounded px-1 font-semibold hover:bg-amber-100 dark:hover:bg-amber-900"
          >
            ×
          </button>
        </div>
      )}

      {showTour && !gateOpen && (
        <GuidedTour
          step={tourStep}
          onPrev={() => setTourStep((s) => Math.max(0, s - 1))}
          onNext={() => setTourStep((s) => Math.min(TOUR_STEPS.length - 1, s + 1))}
          onSkip={closeTour}
          onStart={() => {
            closeTour()
            sendCommand("commencer le diagnostic")
          }}
        />
      )}

      <div className="flex min-h-0 w-full flex-1 flex-col md:flex-row">
        <div
          className={
            showRightPane
              ? "min-h-0 flex-1 basis-1/2 border-b border-slate-200 md:border-b-0 md:border-r dark:border-slate-700"
              : "min-h-0 w-full flex-1"
          }
        >
          {gateOpen ? (
            <div className="flex h-full items-center justify-center p-6">
              <form
                onSubmit={submitAccessCode}
                className="w-full max-w-xs space-y-3 rounded-2xl border border-slate-200 bg-white p-5 shadow-lg dark:border-slate-700 dark:bg-slate-800"
              >
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Code d'accès formateur
                </h2>
                <p className="text-xs text-slate-600 dark:text-slate-300">
                  Cet espace de test est réservé aux formateurs. Saisissez le code qui vous a été communiqué.
                </p>
                {accessMessage && (
                  <p className="text-xs text-red-600 dark:text-red-400">{accessMessage}</p>
                )}
                <input
                  type="password"
                  value={codeInput}
                  onChange={(e) => setCodeInput(e.target.value)}
                  autoFocus
                  autoComplete="off"
                  placeholder="Code d'accès"
                  className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                />
                <button
                  type="submit"
                  className="w-full rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200"
                >
                  Entrer
                </button>
              </form>
            </div>
          ) : accessRequired === null ? (
            <div className="flex h-full items-center justify-center p-6 text-xs text-slate-500 dark:text-slate-400">
              Connexion au serveur…
            </div>
          ) : (
            <ChatKit control={chatkit.control} className="block h-full w-full" />
          )}
        </div>

        {showRightPane && (
          <div className="flex min-h-0 flex-1 basis-1/2 flex-col bg-slate-100 dark:bg-slate-950">
            {/* Header */}
            <div className="flex shrink-0 items-center justify-between px-2 py-1 text-xs text-slate-600 dark:text-slate-300">
              <span className="min-w-0 truncate">
                {hasPdf && "Source PDF"}
                {hasPlot && "Graphique"}
                {!hasPlot && hasMap && mapState && (
                  <>
                    Carte : {mapState.lat.toFixed(4)}, {mapState.lng.toFixed(4)} (zoom {mapState.zoom})
                  </>
                )}
              </span>
              <button
                type="button"
                className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-slate-900/90 px-3 py-1 text-[11px] font-medium text-slate-50 shadow-sm ring-1 ring-slate-700/70 transition-colors hover:bg-slate-900 hover:ring-slate-500/80 dark:bg-slate-800/90 dark:text-slate-100 dark:ring-slate-600/70 dark:hover:bg-slate-700"
                onClick={() => {
                  setMapState(null)
                  setPlotHtml(null)
                  setPdfUrl(null)
                }}
              >
                <span className="text-[13px] leading-none">×</span>
                <span>Fermer</span>
              </button>
            </div>

            {/* Content: fills remaining height */}
            <div className="min-h-0 flex-1">
              {hasPdf ? (
                <iframe
                  title="PDF source"
                  src={pdfUrl ?? ""}
                  className="h-full w-full border-0 bg-white"
                />
              ) : hasPlot ? (
                <iframe
                  title="Graphique"
                  srcDoc={plotHtml ?? ""}
                  className="h-full w-full border-0 bg-white"
                  sandbox="allow-scripts allow-same-origin"
                />
              ) : hasMap ? (
                <iframe
                  title="Carte OpenStreetMap"
                  src={mapUrl as string}
                  className="h-full w-full border-0"
                  loading="lazy"
                />
              ) : null}
            </div>
          </div>
        )}
      </div>

      {scriptError && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/95 p-6 text-center backdrop-blur-sm dark:bg-slate-900/95">
          <div className="max-w-xs space-y-3">
            <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
              Le chat n&apos;a pas pu se charger.
            </p>
            <p className="break-words text-xs text-slate-500 dark:text-slate-400">{scriptError}</p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center rounded-full bg-slate-900 px-4 py-1.5 text-xs font-medium text-white hover:opacity-90 dark:bg-slate-100 dark:text-slate-900"
            >
              Réessayer
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatKitComponent
