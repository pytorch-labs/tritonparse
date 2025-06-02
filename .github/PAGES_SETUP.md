# GitHub Pages Setup for TritonParse

This document explains how to set up automatic deployment to GitHub Pages for the TritonParse website.

## Prerequisites

1. Enable GitHub Pages in your repository settings
2. Set the source to "GitHub Actions" (not "Deploy from a branch")

## Workflow Options

### Option 1: Standard Deployment (Recommended)

- **File**: `.github/workflows/deploy-pages.yml`
- **Trigger**: Automatic on push to `main` branch when website files change
- **Output**: Standard Vite build with separate CSS/JS files
- **Use case**: Standard web hosting with multiple files

### Option 2: Standalone Deployment

- **File**: `.github/workflows/deploy-pages-standalone.yml`
- **Trigger**: Manual dispatch only
- **Output**: Single HTML file with all assets inlined
- **Use case**: Single-file deployment, easier hosting on simple servers

## Setup Steps

### 1. Repository Settings

1. Go to your repository → Settings → Pages
2. Under "Source", select "GitHub Actions"
3. Save the settings

### 2. Workflow Files

The workflow files are already created in `.github/workflows/`:

- `deploy-pages.yml` - Standard deployment
- `deploy-pages-standalone.yml` - Standalone deployment

### 3. Branch Protection (Optional but Recommended)

1. Go to Settings → Branches
2. Add a branch protection rule for `main`
3. Require status checks to pass before merging
4. Include the "build-and-deploy" check

## How It Works

### Standard Deployment Workflow:

1. Triggered when files in `website/` directory are pushed to `main`
2. Sets up Node.js 22
3. Installs dependencies with `npm ci`
4. Builds the website with `npm run build`
5. Uploads the `dist` folder to GitHub Pages
6. Deploys automatically

### Standalone Deployment Workflow:

1. Must be triggered manually from the Actions tab
2. Builds using `npm run build:single`
3. Creates a single `index.html` file with all assets inlined
4. Deploys the standalone file

## Manual Deployment

To manually trigger a deployment:

1. Go to Actions tab in your repository
2. Select the workflow you want to run
3. Click "Run workflow"
4. Select the branch (usually `main`)
5. Click "Run workflow"

## Troubleshooting

### Common Issues:

1. **"Pages not enabled"**: Enable Pages in repository settings and set source to "GitHub Actions"

2. **"Permission denied"**: Ensure the repository has the necessary permissions:

   - Go to Settings → Actions → General
   - Under "Workflow permissions", select "Read and write permissions"

3. **"Build fails"**: Check the Actions tab for detailed error logs

   - Common causes: Missing dependencies, Node.js version mismatch

4. **"404 after deployment"**:
   - Check if the base URL is correct in `vite.config.ts`
   - Ensure the `base: './'` setting is appropriate for your GitHub Pages URL

### Build Configuration

The website is configured with:

- **Base URL**: `./` (relative paths for GitHub Pages compatibility)
- **Output format**: IIFE (Immediately Invoked Function Expression)
- **Asset naming**: Includes hash for cache busting

## Environment Variables

No special environment variables are needed for the basic setup. The workflows use:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- Node.js 22: Specified in the workflow

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to the `website/public/` directory
2. Put your domain name in the file (e.g., `tritonparse.example.com`)
3. Configure DNS settings with your domain provider
4. Enable "Enforce HTTPS" in Pages settings

## Monitoring

- Check the Actions tab to monitor deployment status
- GitHub Pages deployments are visible in the repository's Environments section
- Each successful deployment will show the live URL

## Local Testing

Before deploying, test locally:

```bash
cd website
npm install
npm run build
npm run preview
```

For standalone version:

```bash
npm run build:single
# Open dist/standalone.html in a browser
```
