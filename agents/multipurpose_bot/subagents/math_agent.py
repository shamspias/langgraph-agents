"""
Math Agent for handling mathematical calculations
"""
from typing import Dict, Any, Type
import re
import ast
import operator
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from core.base_agent import BaseAgent
from core.state import MathCalculationState, InputState, OutputState, MathFormula


class MathCalculator:
    """Safe mathematical expression evaluator"""

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    # Allowed functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """
        Safely evaluate a mathematical expression

        Args:
            expression: Mathematical expression as string

        Returns:
            Result of the calculation

        Raises:
            ValueError: If expression is invalid or contains unsafe operations
        """
        try:
            # Parse the expression
            node = ast.parse(expression, mode='eval')

            # Evaluate the AST
            return cls._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    @classmethod
    def _eval_node(cls, node):
        """Recursively evaluate an AST node"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Fallback for older Python
            return node.n
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            operator_func = cls.OPERATORS.get(type(node.op))
            if operator_func:
                return operator_func(left, right)
            else:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            operator_func = cls.OPERATORS.get(type(node.op))
            if operator_func:
                return operator_func(operand)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func = cls.FUNCTIONS.get(node.func.id)
                if func:
                    args = [cls._eval_node(arg) for arg in node.args]
                    return func(*args)
            raise ValueError(f"Unsupported function: {node.func}")
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")


class MathAgent(BaseAgent):
    """Agent for handling mathematical calculations"""

    def __init__(self, **kwargs):
        """Initialize Math Agent"""
        super().__init__(
            name="MathAgent",
            system_prompt=self._get_math_prompt(),
            **kwargs
        )
        self.calculator = MathCalculator()
        self.formula_extractor = self.model.with_structured_output(MathFormula)

    def _get_math_prompt(self) -> str:
        """Get specialized system prompt for math"""
        return """You are an expert mathematician assistant. Your role is to:
        1. Extract mathematical expressions from natural language
        2. Convert them into proper mathematical formulas
        3. Calculate results accurately
        4. Explain calculations step by step

        Always show your work and provide clear explanations.
        Format mathematical expressions clearly using standard notation.
        """

    def get_state_schema(self) -> Type:
        """Get the state schema for this agent"""
        return MathCalculationState

    def build_graph(self) -> StateGraph:
        """Build the math agent graph"""
        workflow = StateGraph(
            MathCalculationState,
            input=InputState,
            output=OutputState
        )

        # Add nodes
        workflow.add_node("extract_formula", self.extract_formula)
        workflow.add_node("calculate", self.calculate_expression)
        workflow.add_node("explain_steps", self.explain_calculation)

        # Add edges
        workflow.add_edge(START, "extract_formula")
        workflow.add_edge("extract_formula", "calculate")
        workflow.add_edge("calculate", "explain_steps")
        workflow.add_edge("explain_steps", END)

        return workflow

    def extract_formula(self, state: MathCalculationState, config: RunnableConfig) -> Dict[str, Any]:
        """Extract mathematical formula from query"""
        self.logger.info("Extracting mathematical formula")

        messages = state["messages"]
        query = messages[-1].content if messages else ""

        # Extract formula
        extraction_prompt = """Extract the mathematical expression from this query.
        Convert any words to mathematical operators:
        - "plus", "add" -> +
        - "minus", "subtract" -> -
        - "times", "multiply" -> *
        - "divided by", "over" -> /
        - "power", "to the" -> **
        - "modulo", "mod" -> %

        Query: {query}

        Return the pure mathematical expression that can be calculated."""

        formula_result = self.formula_extractor.invoke([
            SystemMessage(content=extraction_prompt.format(query=query))
        ])

        # Clean the formula
        formula = self._clean_formula(formula_result.formula if formula_result.formula else query)

        return {
            "expression": query,
            "formula": formula,
            "metadata": {
                "has_math": formula_result.has_math,
                "variables": formula_result.variables
            }
        }

    def _clean_formula(self, formula: str) -> str:
        """Clean and standardize mathematical formula"""
        if not formula:
            return ""

        # Replace common words with operators
        replacements = {
            ' plus ': '+',
            ' add ': '+',
            ' minus ': '-',
            ' subtract ': '-',
            ' times ': '*',
            ' multiply ': '*',
            ' multiplied by ': '*',
            ' divided by ': '/',
            ' over ': '/',
            ' to the power of ': '**',
            ' squared': '**2',
            ' cubed': '**3',
            ' mod ': '%',
            ' modulo ': '%',
        }

        formula_lower = formula.lower()
        for word, op in replacements.items():
            formula_lower = formula_lower.replace(word, op)

        # Extract just the mathematical expression
        # Look for patterns like "2+2", "3*4-5", etc.
        math_pattern = r'[\d\s\+\-\*\/\(\)\.\%\*\*]+'
        matches = re.findall(math_pattern, formula_lower)

        if matches:
            # Take the longest match
            formula = max(matches, key=len)
            # Remove extra spaces
            formula = re.sub(r'\s+', ' ', formula).strip()
            formula = formula.replace(' ', '')

        return formula

    def calculate_expression(self, state: MathCalculationState, config: RunnableConfig) -> Dict[str, Any]:
        """Calculate the mathematical expression"""
        self.logger.info("Calculating expression")

        formula = state.get("formula", "")

        if not formula:
            return {
                "error": "No mathematical expression found",
                "result": None
            }

        try:
            # Calculate using safe evaluator
            result = self.calculator.evaluate(formula)

            self.logger.info(f"Calculated: {formula} = {result}")

            return {
                "result": result,
                "metadata": {
                    **state.get("metadata", {}),
                    "calculation_success": True
                }
            }
        except Exception as e:
            self.logger.error(f"Calculation error: {str(e)}")
            return {
                "error": f"Calculation error: {str(e)}",
                "result": None
            }

    def explain_calculation(self, state: MathCalculationState, config: RunnableConfig) -> Dict[str, Any]:
        """Explain the calculation step by step"""
        self.logger.info("Explaining calculation")

        expression = state.get("expression", "")
        formula = state.get("formula", "")
        result = state.get("result")
        error = state.get("error")

        if error:
            response = f"I encountered an error with your calculation: {error}\n"
            response += "Please check your expression and try again."
        elif result is not None:
            # Generate step-by-step explanation
            steps = self._generate_steps(formula)

            response = f"Let me calculate: {expression}\n\n"
            response += f"Mathematical expression: {formula}\n\n"

            if steps:
                response += "Step-by-step solution:\n"
                for i, step in enumerate(steps, 1):
                    response += f"{i}. {step}\n"

            response += f"\n**Result: {result}**"

            # Format result nicely
            if isinstance(result, float):
                if result.is_integer():
                    response += f"\n\nThe answer is {int(result)}"
                else:
                    response += f"\n\nThe answer is {result:.4f}"
            else:
                response += f"\n\nThe answer is {result}"
        else:
            response = "I couldn't find a mathematical expression in your query. "
            response += "Please provide a calculation for me to solve."

        return {
            "messages": [AIMessage(content=response)],
            "response": response,
            "step_by_step": steps if result is not None else [],
            "metadata": state.get("metadata", {})
        }

    def _generate_steps(self, formula: str) -> list:
        """Generate step-by-step explanation of the calculation"""
        steps = []

        try:
            # Parse order of operations
            if '**' in formula:
                steps.append(f"Calculate exponents in: {formula}")
            if '*' in formula or '/' in formula or '%' in formula:
                steps.append(f"Perform multiplication/division from left to right")
            if '+' in formula or '-' in formula:
                steps.append(f"Perform addition/subtraction from left to right")

            # Add the formula itself as a step
            steps.append(f"Calculate: {formula}")

        except Exception as e:
            self.logger.error(f"Error generating steps: {str(e)}")

        return steps
