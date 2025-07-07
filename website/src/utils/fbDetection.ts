/**
 * Utility to detect if the fb directory exists
 * This checks for the existence of the exists marker file in the fb directory
 */

export const checkFbDirectoryExists = async (): Promise<boolean> => {
  try {
    // Try to fetch the exists marker file in the fb directory
    const response = await fetch('/fb/exists', { method: 'HEAD' });
    return response.ok;
  } catch (error) {
    // If the request fails, the directory likely doesn't exist
    return false;
  }
}; 