const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const ts = require('typescript')
const output = ts.transpileModule(fs.readFileSync(`${__dirname}/src/app/timedFetch.ts`, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText
const exportsObject = {}
new Function('exports', output)(exportsObject)
const { timedFetch } = exportsObject

test('deadline covers headers and stalled streamed bodies; normal responses work', async () => {
  const original = global.fetch
  try {
    global.fetch = (_, { signal }) => new Promise((resolve, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    })
    await assert.rejects(timedFetch('http://test', {}, 20), /trop de temps/)
    global.fetch = async (_, { signal }) => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('partial'))
        signal.addEventListener('abort', () => controller.error(signal.reason), { once: true })
      },
    }))
    const response = await timedFetch('http://test', {}, 20)
    await assert.rejects(response.text(), /trop de temps/)
    global.fetch = async () => new Response('complete')
    assert.equal(await (await timedFetch('http://test', {}, 100)).text(), 'complete')
  } finally { global.fetch = original }
})
