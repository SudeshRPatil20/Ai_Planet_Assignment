/**
 * Math API Module
 * 
 * This module handles all API communications with the FastAPI backend.
 * It includes proper error handling, type safety, and timeout management
 * for academic-quality code standards.
 */

import type { MathSolution, MathRequest } from '../types/mathTypes';

// API Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT = 30000; // 30 seconds timeout for complex math problems

/**
 * Custom error class for API-related errors
 */
export class ApiError extends Error {
  public status?: number;
  public response?: Response;

  constructor(message: string, status?: number, response?: Response) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.response = response;
  }
}

/**
 * Timeout controller to cancel hanging requests
 */
const createTimeoutController = (timeout: number): AbortController => {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), timeout);
  return controller;
};

/**
 * Input validator
 */
const validateMathQuestion = (question: string): void => {
  if (!question || typeof question !== 'string') {
    throw new Error('Question must be a non-empty string');
  }
  if (question.trim().length === 0) {
    throw new Error('Question cannot be empty');
  }
  if (question.length > 1000) {
    throw new Error('Question is too long (maximum 1000 characters)');
  }
};

/**
 * Validate backend response
 */
const validateApiResponse = (data: any): data is MathSolution => {
  console.log("kp", data, "type======", typeof data)
  if (!data || typeof data !== 'object') return false;
  // if (typeof data.answer !== 'string' || !data.answer.trim()) return false;
  if (!Array.isArray(data.steps)) return false;
  // if (typeof data.topic !== 'string' || !data.topic.trim()) return false;
  // if (typeof data.difficulty !== 'string' || !data.difficulty.trim()) return false;
  if (!data.steps.every((step: any) => typeof step === 'string')) return false;
  return true;
};

/**
 * Main function to solve math problems via FastAPI backend
 */
export const solveMathProblem = async (
  question: string,
  userId: string = 'anonymous'
): Promise<MathSolution> => {
  validateMathQuestion(question);

  const requestPayload: MathRequest = {
    question: question.trim(),
    user_id: userId
  };

  console.log(requestPayload)

  const controller = createTimeoutController(REQUEST_TIMEOUT);

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(requestPayload),
      signal: controller.signal,
    });
console.log(response.ok)
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = Array.isArray(errorData.detail)
            ? errorData.detail.map((err: any) => err.msg || err).join(', ')
            : errorData.detail;
        }
      } catch {}
      throw new ApiError(errorMessage, response.status, response);
    }

    let responseData;
    try {
      responseData = await response.json();
      console.log("sp",responseData)
    } catch {
      throw new ApiError('Invalid JSON response from server');
    }
    //  console.log(responseData)
    if (!validateApiResponse(responseData)) {
      throw new ApiError('Invalid response format from server');
    }
   console.log("response data",responseData)
    return responseData as MathSolution;

  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new ApiError('Unable to connect to the math tutor service. Please check if the backend is running.');
    }
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('Request timed out. The problem might be too complex or the server is busy.');
    }
    throw new ApiError(`Unexpected error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

/**
 * Ping the backend to check health
 */
export const checkBackendHealth = async (): Promise<boolean> => {
  const controller = createTimeoutController(5000);
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  }
};
