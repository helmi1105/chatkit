"use client"

import { ColorScheme } from "@openai/chatkit-react"
import ChatKitComponent from "./ChatKitComponent"
import { useState, useEffect, type FormEvent } from "react"

type Account = { user_id: string; username: string }
const THEME_STORAGE_KEY = "chatkit-theme"

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStorage(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value)
  } catch (error) {
    console.error("Failed to persist", key, error)
  }
}

export default function Home() {
  const [account, setAccount] = useState<Account | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [registering, setRegistering] = useState(false)
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [theme, setTheme] = useState<ColorScheme>("light")

  useEffect(() => {
    fetch('/backend/auth/me', { credentials: 'same-origin', cache: 'no-store' })
      .then(async response => {
        if (response.ok) setAccount(await response.json())
        else if (response.status !== 401) setError('Impossible de joindre le serveur.')
      })
      .catch(() => setError('Impossible de joindre le serveur.'))
      .finally(() => setLoading(false))
    const expired = () => {
      setAccount(null)
      setPassword("")
      setError('Votre session a expiré. Veuillez vous reconnecter.')
    }
    window.addEventListener('chatkit-session-expired', expired)
    const storedTheme = readStorage(THEME_STORAGE_KEY)
    if (storedTheme === "dark" || storedTheme === "light") {
      setTheme(storedTheme)
    }
    return () => window.removeEventListener('chatkit-session-expired', expired)
  }, [])

  const login = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError("")
    try {
      const response = await fetch(registering ? '/backend/auth/register' : '/backend/auth/login', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-ChatKit-Request': '1' },
        body: JSON.stringify({ username, password }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.message ?? 'Connexion impossible.')
      setAccount(body)
      setPassword("")
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Connexion impossible.')
    } finally { setBusy(false) }
  }

  const logout = async () => {
    setBusy(true)
    setError("")
    try {
      const response = await fetch('/backend/auth/logout', {
        method: 'POST', credentials: 'same-origin', headers: { 'X-ChatKit-Request': '1' },
      })
      if (!response.ok && response.status !== 401) throw new Error('Déconnexion impossible. Réessayez.')
      window.sessionStorage.removeItem('chatkit-provider-api-key')
      setAccount(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Déconnexion impossible.')
    } finally { setBusy(false) }
  }

  const toggleTheme = () => {
    setTheme((v) => {
      const next: ColorScheme = v === "dark" ? "light" : "dark"
      writeStorage(THEME_STORAGE_KEY, next)
      return next
    })
  }

  return (
    <main
      className={`flex min-h-screen flex-col items-center bg-slate-100 dark:bg-slate-950 ${
        theme === "dark" ? "dark" : ""
      }`}
    >
      <div className="mx-auto flex h-[100dvh] w-full max-w-4xl flex-col px-2 py-2 sm:px-4">
        <div className="flex w-full flex-row flex-wrap items-center justify-end gap-2 pb-2">
          {account && <>
            <span className="text-sm dark:text-white">{account.username}</span>
            <button type="button" disabled={busy} onClick={logout} className="rounded-lg border px-3 py-1.5 text-sm dark:text-white">Se déconnecter</button>
          </>}
          <button
            type="button"
            className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200"
            onClick={toggleTheme}
          >
            {theme === "dark" ? "Mode clair" : "Mode sombre"}
          </button>
        </div>
        {error && <p role="alert" className="my-3 text-sm text-red-600">{error}</p>}
        {loading ? <p role="status">Chargement…</p> : account ? (
          <ChatKitComponent key={account.user_id} userId={account.user_id} theme={theme} />
        ) : (
          <form onSubmit={login} className="m-auto w-full max-w-sm space-y-5 rounded-2xl bg-white p-8 shadow-sm dark:bg-slate-900 dark:text-white">
            <h1 className="text-2xl font-semibold">{registering ? 'Créer votre compte' : 'Votre espace d’apprentissage'}</h1>
            <p className="text-sm text-slate-500">{registering ? 'Choisissez vos identifiants pour commencer votre apprentissage.' : 'Connectez-vous pour retrouver vos échanges et votre progression.'}</p>
            <label className="block text-sm">Nom d’utilisateur
              <input required autoComplete="username" minLength={registering ? 3 : undefined} maxLength={64} pattern={registering ? '[a-zA-Z0-9][a-zA-Z0-9_.\\-]{2,63}' : undefined} value={username} onChange={e => setUsername(e.target.value)} className="mt-2 w-full rounded-lg border p-2 dark:bg-slate-800" />
            </label>
            <label className="block text-sm">Mot de passe
              <input required type="password" autoComplete={registering ? 'new-password' : 'current-password'} minLength={registering ? 4 : undefined} maxLength={256} value={password} onChange={e => setPassword(e.target.value)} className="mt-2 w-full rounded-lg border p-2 dark:bg-slate-800" />
            </label>
            {registering && <p className="text-xs text-slate-500">Nom : 3 à 64 lettres ou chiffres, avec ., _ ou -. Mot de passe : au moins 4 caractères (lettres, chiffres ou les deux).</p>}
            <button disabled={busy} className="w-full rounded-lg bg-blue-700 p-3 text-white disabled:opacity-50">{busy ? 'Un instant…' : registering ? 'Créer mon compte' : 'Se connecter'}</button>
            <button type="button" disabled={busy} onClick={() => { setRegistering(!registering); setPassword(''); setError('') }} className="w-full rounded-lg border p-3 text-sm disabled:opacity-50">
              {registering ? 'Déjà un compte ? Se connecter' : 'Créer un compte'}
            </button>
          </form>
        )}
      </div>
    </main>
  )
}
