import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const envDir = path.resolve(__dirname, '../settings')
  const settingsEnv = loadEnv(mode, envDir, '')
  const localEnv = loadEnv(mode, __dirname, '')
  const env = { ...localEnv, ...settingsEnv }

  const fastapiHost = env.FASTAPI_HOST || '127.0.0.1'
  const fastapiPort = Number(env.FASTAPI_PORT || 8000)
  const uiHost = env.UI_HOST || '127.0.0.1'
  const uiPort = Number(env.UI_PORT || 5173)
  const apiBase = env.VITE_API_BASE_URL || '/api'
  const apiBasePath = apiBase.startsWith('/') ? apiBase : `/${apiBase}`
  const apiBaseRegex = new RegExp(
    `^${apiBasePath.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')}`,
  )

  const proxyTarget = `http://${fastapiHost}:${fastapiPort}`

  return {
    envDir,
    plugins: [react()],
    server: {
      host: uiHost,
      port: uiPort,
      strictPort: false,
      proxy: {
        [apiBasePath]: {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(apiBaseRegex, ''),
          // 10 minute timeout for large dataset downloads
          timeout: 600000,
        },
      },
    },
    preview: {
      host: uiHost,
      port: uiPort,
      strictPort: false,
      proxy: {
        [apiBasePath]: {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(apiBaseRegex, ''),
          // 10 minute timeout for large dataset downloads
          timeout: 600000,
        },
      },
    },
  }
})
