from typing import Dict, Any, Optional
from datetime import datetime
from .reasoning import generate_text_response

class ExplainabilityGenerator:
    """
    Generates a CFO-safe, plain English explanation for R&D tax credit projects.
    Focuses on business impact and IRS vocabulary without technical or AI jargon.
    """
    
    def __init__(self):
        pass

    def generate_cfo_explanation(
        self, 
        project_name: str, 
        project_description: str, 
        eligible: bool, 
        rationale: str,
        confidence: float = 0.0,
        total_qre: float = 0.0
    ) -> str:
        """
        Generate a structured 1-page narrative for non-technical stakeholders (CFO/Audit).
        """
        
        system_prompt = (
            "You are a senior R&D Tax Credit consultant specializing in audit defense and executive communication. "
            "Your user is a CFO who is non-technical and risk-averse. "
            "Your task is to provide a decision recommendation for R&D eligibility."
        )
        
        prompt = f"""
        **Objective**: Provide a structured "CFO-Safe" decision summary for the following project.
        
        **Constraints**:
        1. **Format**: Use the exact format below.
        2. **Tone**: Professional but guarded. Use ambiguous language (e.g., "likely eligible", "appears to meet", "suggests alignment") instead of absolute certainty (e.g., "definitely", "guaranteed").
        3. **Plain English**: No engineering or AI jargon.
        4. **Length**: Concise explanation (approx 150-200 words).
        
        **Project Details**:
        - Name: {project_name}
        - Description: {project_description}
        - Computed Eligibility: {"Eligible" if eligible else "Not Eligible"}
        - Computed Confidence: {confidence:.2f}
        - Key Technical Indicators: {rationale}
        - Estimated QRE: ${total_qre:,.2f}
        
        **Output Format (Markdown)**:
        
        # Recommendation: [Approve / Reject]
        
        **Confidence Score**: [Low / Medium / High] (based on computed confidence {confidence:.2f})
        
        **Explanation**:
        [Detailed reason why this project should be approved or rejected. Map the activities to IRS criteria (Business Component, Uncertainty, Experimentation) but explain them simply. Use phrases like "The documentation suggests..." or "It appears that...".]
        
        """
        
        explanation = generate_text_response(system_prompt, prompt)
        return explanation
        
        explanation = generate_text_response(system_prompt, prompt)
        return explanation
