import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'fs'
import { resolve } from 'path'
import { execSync } from 'child_process'

const packageJson = JSON.parse(
  readFileSync(resolve(__dirname, 'package.json'), 'utf-8')
)

// Build-time metadata
const buildDate = process.env.BUILD_DATE || new Date().toISOString()
let gitSha = process.env.GIT_COMMIT_SHA_SHORT
if (!gitSha) {
  try {
    gitSha = execSync('git rev-parse --short HEAD').toString().trim()
  } catch {
    gitSha = 'unknown'
  }
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'strip-module-attr',
      enforce: 'post',
      apply: 'build', // only apply this plugin during build
      transformIndexHtml(html) {
        return html.replace(
          /<script\s+type=["']module["']\s+([^>]*?)src=/g,
          '<script defer $1src='
        )
      }
    }
  ],

  base: './',
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        format: 'iife',
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    }
  },
  define: {
    'import.meta.env.PACKAGE_VERSION': JSON.stringify(packageJson.version),
    'import.meta.env.PACKAGE_BUILD_DATE': JSON.stringify(buildDate),
    'import.meta.env.GIT_COMMIT_SHA_SHORT': JSON.stringify(gitSha)
  }
})
