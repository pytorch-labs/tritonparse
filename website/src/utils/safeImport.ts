/**
 * Safe dynamic import utility that handles missing modules gracefully
 * This avoids build-time module resolution issues
 */

/**
 * Safely imports a module by path, returning null if it doesn't exist
 * Uses Function constructor to avoid static analysis
 */
export async function safeImport(modulePath: string): Promise<any | null> {
  try {
    // Use Function constructor to create dynamic import
    // This completely bypasses static analysis
    const dynamicImport = new Function('path', 'return import(path)');
    const module = await dynamicImport(modulePath);
    return module;
  } catch (error) {
    console.warn(`Module ${modulePath} not available:`, error);
    return null;
  }
} 