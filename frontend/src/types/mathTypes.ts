/**
 * Type Definitions for Math Tutor Agent
 *
 * This file contains all TypeScript interfaces and types used throughout
 * the application. It ensures type safety and better code documentation
 * for academic-quality standards.
 */

/**
 * Request payload structure for the FastAPI backend
 * Used when sending math problems to the /query endpoint
 */
export interface MathRequest {
  /** The math problem/question to be solved */
  question: string;

  /** The user identifier for tracking or personalization */
  user_id: string;
}

/**
 * Response structure from the FastAPI backend
 * Contains the complete solution with all necessary details
 */
export interface MathSolution {
  /** The final answer to the math problem */
  answer: string;

  /** Array of step-by-step solution explanations */
  steps: string[];

  /** Optional: The mathematical topic/category (e.g., "Algebra", "Calculus") */
  topic?: string;

  /** Optional: Difficulty level of the problem (e.g., "Easy", "Medium", "Hard") */
  difficulty?: DifficultyLevel;

  /** Optional: Source from where the solution was derived */
  source?: string;

  /** Optional: Confidence score between 0 and 1 */
  confidence?: number;

  /** Optional: How the query was processed (e.g., "web_search", "symbolic") */
  query_type?: string;

  /** Optional: External sources used to generate the solution */
  sources_used?: string[];
}

/**
 * Possible difficulty levels for math problems
 * Used for consistent difficulty classification
 */
export type DifficultyLevel = 'Easy' | 'Medium' | 'Hard';

/**
 * Common mathematical topics supported by the system
 * Extensible list for future topic additions
 */
export type MathTopic =
  | 'Algebra'
  | 'Geometry'
  | 'Calculus'
  | 'Trigonometry'
  | 'Statistics'
  | 'Linear Algebra'
  | 'Differential Equations'
  | 'Number Theory'
  | 'Graph Theory'
  | 'Other';

/**
 * Application state interface for the main component
 * Helps manage complex state in React components
 */
export interface AppState {
  /** Current math question being processed */
  currentQuestion: string;

  /** Current solution (null if no solution yet) */
  currentSolution: MathSolution | null;

  /** Loading state for API requests */
  isLoading: boolean;

  /** Error message (empty string if no error) */
  errorMessage: string;

  /** Whether the solution steps are expanded */
  isStepsExpanded: boolean;
}

/**
 * Props interface for reusable solution display component
 * Enables component reusability and prop validation
 */
export interface SolutionDisplayProps {
  /** The math solution to display */
  solution: MathSolution;

  /** Whether steps should be initially expanded */
  initiallyExpanded?: boolean;

  /** Optional callback when user interacts with the solution */
  onInteraction?: (action: string) => void;
}

/**
 * Configuration interface for API settings
 * Allows for flexible API configuration
 */
export interface ApiConfig {
  /** Base URL for the FastAPI backend */
  baseUrl: string;

  /** Request timeout in milliseconds */
  timeout: number;

  /** Maximum retries for failed requests */
  maxRetries: number;
}
