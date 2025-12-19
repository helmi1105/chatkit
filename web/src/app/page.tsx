"use client"

import { ColorScheme } from "@openai/chatkit-react"
import ChatKitComponent from "./ChatKitComponent"
import { useState, useEffect } from 'react'

const USERID_STORAGE_KEY = "userId"

export default function Home() {
    // A free-form string that identifies the end user
    // ensures this Session can access other objects that have the same user scope.
    const [userId, setUserId] = useState<string | null>(null)
    const [theme, setTheme] = useState<ColorScheme>("dark")
    const [maximize, setMax] = useState<boolean>(true)

    function readUserId(): string | null {
        if (typeof window === "undefined") {
            return null
        }
        try {
            return window.localStorage.getItem(USERID_STORAGE_KEY)
        } catch (error) {
            return null
        }
    }

    function persistUserId(id: string): void {
        if (typeof window === "undefined") {
            return
        }
        try {
            window.localStorage.setItem(USERID_STORAGE_KEY, id)
        } catch (error) {
            console.error("Failed to persist userid", error)
        }
    }

    useEffect(() => {
        if (typeof window === "undefined") {
            return
        }

        const existing = readUserId()
        console.log("exisitng: ", existing)
        if (existing === null) {
            const id = crypto.randomUUID()
            setUserId(id)
            persistUserId(id)
        } else {
            setUserId(existing)
        }
    }, [(typeof window)])

    return (
        <main className="flex min-h-screen flex-col items-center justify-end bg-slate-100 dark:bg-slate-950">
            <div className="m-auto w-full h-[90vh] max-w-4xl flex flex-col">
                <div className="justify-end w-full flex flex-row py-2 gap-2">
                    <button
                        type="button"
                        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200 max-w-fit pointer-events-auto"
                        onClick={() => setMax((v) => {
                            return !v
                        })}>
                        {maximize ? "Right Pane" : "Maximize"}
                    </button>

                    <button
                        type="button"
                        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200 max-w-fit pointer-events-auto"
                        onClick={() => setTheme((v) => {
                            return v === "dark" ? "light" : "dark"
                        })}>
                        {theme === "dark" ? "Use Light" : "Use Dark"}
                    </button>
                </div>
                {
                    userId === null ? null : <ChatKitComponent
                        userId={userId}
                        theme={theme}
                        maximize={maximize}
                        onResponseEnd={() => {
                            console.log("Response finished")
                        }}
                    />
                }

            </div>
        </main>
    )
}
