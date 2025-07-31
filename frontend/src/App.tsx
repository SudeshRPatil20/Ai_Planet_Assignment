import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calculator,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  BookOpen,
  Sparkles,
  Star,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { solveMathProblem } from "./api/mathApi";
import type { MathSolution } from "./types/mathTypes";
import "./App.css";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css"; // Make sure this is imported once globally

/**
 * Main Math Tutor Agent Application Component
 *
 * Modern, dark-themed interface inspired by contemporary SaaS applications
 * Features smooth animations, gradient backgrounds, and professional styling
 */
function App() {
  // State management for the application
  const [question, setQuestion] = useState<string>("");
  const [solution, setSolution] = useState<MathSolution | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [isStepsExpanded, setIsStepsExpanded] = useState<boolean>(false);

  /**
   * Handles form submission and API communication
   * Includes proper error handling and loading state management
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Input validation
    if (!question.trim()) {
      setError("Please enter a math question");
      return;
    }

    // Reset previous state
    setError("");
    setSolution(null);
    setIsLoading(true);
    setIsStepsExpanded(false);

    try {
      // Make API call to FastAPI backend
      const result = await solveMathProblem(question.trim());
      setSolution(result);
    } catch (err) {
      // Handle different types of errors
      if (err instanceof Error) {
        setError(`Failed to solve problem: ${err.message}`);
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Gets appropriate color scheme for difficulty badge
   */
  const getDifficultyColor = (difficulty: string): string => {
    switch (difficulty.toLowerCase()) {
      case "easy":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "medium":
        return "bg-amber-500/20 text-amber-400 border-amber-500/30";
      case "hard":
        return "bg-red-500/20 text-red-400 border-red-500/30";
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30";
    }
  };

  /**
   * Animation variants for Framer Motion
   */
  const containerVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: "easeOut" },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.95, y: 20 },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" },
    },
  };

  const floatingVariants = {
    animate: {
      y: [-10, 10, -10],
      transition: {
        duration: 6,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>

      {/* Floating decorative icons */}
      <motion.div
        className="absolute top-20 left-20 text-emerald-400/20"
        variants={floatingVariants}
        animate="animate"
      >
        <Sparkles className="w-6 h-6" />
      </motion.div>
      <motion.div
        className="absolute top-40 right-32 text-blue-400/20"
        variants={floatingVariants}
        animate="animate"
        transition={{ delay: 2 }}
      >
        <Star className="w-8 h-8" />
      </motion.div>
      <motion.div
        className="absolute bottom-32 left-1/4 text-purple-400/20"
        variants={floatingVariants}
        animate="animate"
        transition={{ delay: 4 }}
      >
        <Calculator className="w-7 h-7" />
      </motion.div>

      <div className="container mx-auto px-4 py-8 max-w-4xl relative z-10">
        {/* Header Section */}
        <motion.div
          className="text-center mb-16"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-emerald-500/20 rounded-2xl blur-xl"></div>
              <div className="relative bg-gradient-to-r from-emerald-500 to-blue-500 p-4 rounded-2xl">
                <Calculator className="h-12 w-12 text-white" />
              </div>
            </div>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent mb-6 leading-tight">
            Math Tutor Agent
          </h1>

          <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed mb-8">
            Get AI-Powered Step-by-Step Solutions to Math Problems
          </p>

          <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span>Instant Solutions</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span>Step-by-Step Explanations</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
              <span>All Math Topics</span>
            </div>
          </div>
        </motion.div>

        {/* Input Form Section */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          transition={{ delay: 0.3 }}
        >
          <Card className="mb-8 bg-gray-800/50 backdrop-blur-xl border-gray-700/50 shadow-2xl">
            <CardHeader className="text-center pb-6">
              <CardTitle className="text-2xl text-white mb-2">
                Ask Your Math Question
              </CardTitle>
              <CardDescription className="text-gray-400 text-lg">
                Enter any math problem and get detailed solutions instantly
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-3">
                  <label
                    htmlFor="math-question"
                    className="text-sm font-medium text-gray-300 block"
                  >
                    Math Problem *
                  </label>
                  <div className="relative">
                    <Input
                      id="math-question"
                      type="text"
                      placeholder="Ask a math question... (e.g., Solve x² - 4 = 0)"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      className="text-lg py-6 px-6 bg-gray-900/50 border-gray-600 focus:border-emerald-500 focus:ring-emerald-500/20 text-white placeholder-gray-500 transition-all duration-200"
                      disabled={isLoading}
                      aria-describedby="question-help"
                      required
                    />
                    <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                      <ArrowRight className="w-5 h-5 text-gray-500" />
                    </div>
                  </div>
                  <p id="question-help" className="text-sm text-gray-500">
                    Examples: "Factor x² + 5x + 6", "Solve 2x + 3 = 7", "Find
                    the derivative of x³"
                  </p>
                </div>

                <Button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  className="w-full py-6 text-lg font-semibold bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600 text-white border-0 transition-all duration-300 transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isLoading ? (
                    <motion.div
                      className="flex items-center"
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                    >
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                      Solving Problem...
                    </motion.div>
                  ) : (
                    <div className="flex items-center justify-center">
                      <span>Get Solution</span>
                      <Sparkles className="w-5 h-5 ml-2" />
                    </div>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </motion.div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              className="mb-6"
            >
              <Alert className="border-red-500/50 bg-red-500/10 backdrop-blur-sm">
                <AlertCircle className="h-4 w-4 text-red-400" />
                <AlertDescription className="text-red-300 font-medium">
                  {error}
                </AlertDescription>
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Solution Display */}
        <AnimatePresence>
          {solution && (
            <motion.div
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
            >
              <Card className="bg-gray-800/50 backdrop-blur-xl border-gray-700/50 shadow-2xl">
                <CardHeader className="pb-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <CardTitle className="text-2xl text-white flex items-center">
                      <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center mr-3">
                        <BookOpen className="w-4 h-4 text-white" />
                      </div>
                      Solution
                    </CardTitle>
                    <div className="flex flex-wrap gap-2">
                      <Badge
                        variant="outline"
                        className="bg-blue-500/20 text-blue-400 border-blue-500/30 px-3 py-1"
                      >
                        {solution.topic}
                      </Badge>
                      {/* <Badge 
                        variant="outline" 
                        className={`px-3 py-1 ${getDifficultyColor(solution.difficulty)}`}
                      >
                        {solution.difficulty}
                      </Badge> */}
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-6">
                  {/* Final Answer Section */}
                  <div className="relative my-12">
                    {/* Glowing background layer */}
                    <div className="absolute inset-0 rounded-xl blur-md bg-gradient-to-r from-emerald-500/10 to-blue-500/10"></div>

                    {/* Main card content */}
                    <div className="relative rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/15 to-blue-500/15 p-6 shadow-lg">
                      {/* Header */}
                      <div className="flex items-center mb-4">
                        <div className="flex items-center justify-center w-8 h-8 mr-3 rounded-full bg-emerald-500/20">
                          <span className="text-sm text-emerald-400 font-bold">
                            ✓
                          </span>
                        </div>
                        <h3 className="text-lg font-semibold text-emerald-400">
                          Final Answer
                        </h3>
                      </div>

                      {/* Answer Content using Markdown with Math */}
                      <div className="prose prose-invert prose-lg max-w-none rounded-lg bg-gray-900/40 p-5 border border-gray-700/40 text-white overflow-x-auto">
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {solution.answer}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                  {/* <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 rounded-xl blur-sm"></div>
                    <div className="relative bg-gradient-to-r from-emerald-500/20 to-blue-500/20 border border-emerald-500/30 rounded-xl p-6">
                      <h3 className="text-lg font-semibold text-emerald-400 mb-4 flex items-center">
                        <div className="w-8 h-8 bg-emerald-500/20 rounded-full flex items-center justify-center mr-3">
                          <span className="text-emerald-400 text-sm">✓</span>
                        </div>
                        Final Answer
                      </h3>
                      <div className="text-xl font-mono text-white bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
                        {solution.answer}
                      </div>
                    </div>
                  </div> */}

                  {/* Step-by-Step Solution */}
                  {solution.steps && solution.steps.length > 0 && (
                    <Collapsible
                      open={isStepsExpanded}
                      onOpenChange={setIsStepsExpanded}
                    >
                      <CollapsibleTrigger asChild>
                        <Button
                          variant="outline"
                          className="w-full justify-between py-6 text-lg font-medium bg-gray-700/30 border-gray-600/50 text-white hover:bg-gray-700/50 transition-all duration-200"
                        >
                          <span className="flex items-center">
                            <BookOpen className="h-5 w-5 mr-3 text-blue-400" />
                            Step-by-Step Solution ({solution.steps.length}{" "}
                            steps)
                          </span>
                          {isStepsExpanded ? (
                            <ChevronUp className="h-5 w-5 text-gray-400" />
                          ) : (
                            <ChevronDown className="h-5 w-5 text-gray-400" />
                          )}
                        </Button>
                      </CollapsibleTrigger>

                      <CollapsibleContent>
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ duration: 0.3 }}
                          className="mt-4 space-y-4"
                        >
                          {solution.steps.map((step, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.1 }}
                              className="flex gap-4 p-4 bg-gray-900/30 rounded-lg border border-gray-700/30"
                            >
                              <div className="flex-shrink-0">
                                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-semibold">
                                  {index + 1}
                                </div>
                              </div>
                              <div className="flex-1">
                                <p className="text-gray-200 leading-relaxed font-mono text-sm sm:text-base">
                                  {step}
                                </p>
                              </div>
                            </motion.div>
                          ))}
                        </motion.div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.footer
          className="mt-16 text-center text-gray-500 text-sm"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          transition={{ delay: 0.6 }}
        >
          <div className="flex items-center justify-center mb-2">
            <div className="w-1 h-1 bg-emerald-500 rounded-full mx-2"></div>
            <p className="text-gray-400">
              Math Tutor Agent - Academic Assignment Project
            </p>
            <div className="w-1 h-1 bg-emerald-500 rounded-full mx-2"></div>
          </div>
          <p className="text-gray-500">
            Powered by AI for educational excellence
          </p>
        </motion.footer>
      </div>
    </div>
  );
}

export default App;
