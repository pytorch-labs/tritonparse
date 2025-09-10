/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly PACKAGE_VERSION: string;
  readonly PACKAGE_BUILD_DATE: string;
  readonly GIT_COMMIT_SHA_SHORT: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
