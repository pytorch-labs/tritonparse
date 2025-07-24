/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 */

/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly PACKAGE_VERSION: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
