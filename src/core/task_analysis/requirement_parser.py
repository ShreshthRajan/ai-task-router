import re
import spacy
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class ParsedRequirement:
    """Structured representation of a parsed requirement."""
    requirement_id: str
    category: str  # functional, non-functional, technical
    priority: str  # must, should, could, won't
    description: str
    acceptance_criteria: List[str]
    dependencies: List[str]
    technical_constraints: List[str]
    estimated_complexity: float
    confidence_score: float

@dataclass
class TaskRequirements:
    """Complete requirements analysis for a task."""
    task_id: str
    requirements: List[ParsedRequirement]
    overall_scope: str  # small, medium, large, epic
    technical_stack: List[str]
    quality_requirements: Dict[str, str]
    constraints: List[str]
    assumptions: List[str]
    risks: List[str]

class RequirementParser:
    """Advanced NLP-based requirement parsing and analysis."""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        
        # Requirement patterns
        self.requirement_patterns = {
            'functional': [
                r'the system (shall|must|should) (.+)',
                r'user (can|should be able to) (.+)',
                r'(implement|add|create) (.+)',
                r'feature (.+) (should|must) (.+)'
            ],
            'non_functional': [
                r'performance (.+)',
                r'security (.+)',
                r'scalability (.+)',
                r'availability (.+)',
                r'reliability (.+)'
            ],
            'technical': [
                r'use (.+) (framework|library|technology)',
                r'integrate with (.+)',
                r'deploy (to|on) (.+)',
                r'support (.+) (browser|platform|device)'
            ]
        }
        
        # Priority indicators
        self.priority_indicators = {
            'must': ['must', 'required', 'critical', 'essential', 'mandatory'],
            'should': ['should', 'important', 'recommended', 'preferred'],
            'could': ['could', 'nice to have', 'optional', 'if time permits'],
            'wont': ['won\'t', 'will not', 'not in scope', 'future release']
        }
        
        # Acceptance criteria patterns
        self.acceptance_patterns = [
            r'given (.+) when (.+) then (.+)',
            r'acceptance criteria:?\s*(.+)',
            r'ac:?\s*(.+)',
            r'definition of done:?\s*(.+)',
            r'dod:?\s*(.+)'
        ]
        
        # Constraint patterns
        self.constraint_patterns = [
            r'constraint:?\s*(.+)',
            r'limitation:?\s*(.+)',
            r'must (use|support|work with) (.+)',
            r'cannot (use|exceed|modify) (.+)',
            r'within (.+) (days|weeks|hours)',
            r'budget of (.+)',
            r'compatible with (.+)'
        ]
    
    def _load_models(self):
        """Load required NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features will be limited.")
            self.nlp = None
    
    async def parse_task_requirements(self, task_data: Dict) -> TaskRequirements:
        """Parse and analyze task requirements comprehensively."""
        
        # Extract text content
        title = task_data.get('title', '')
        description = task_data.get('body', '')
        labels = [label.get('name', '') for label in task_data.get('labels', [])]
        
        full_text = f"{title}\n{description}"
        
        # Parse individual requirements
        requirements = await self._extract_requirements(full_text)
        
        # Analyze overall scope
        scope = self._determine_scope(full_text, requirements)
        
        # Extract technical stack
        tech_stack = self._extract_technical_stack(full_text, labels)
        
        # Parse quality requirements
        quality_reqs = self._extract_quality_requirements(full_text)
        
        # Extract constraints
        constraints = self._extract_constraints(full_text)
        
        # Extract assumptions
        assumptions = self._extract_assumptions(full_text)
        
        # Identify risks
        risks = self._identify_requirement_risks(full_text, requirements)
        
        return TaskRequirements(
            task_id=task_data.get('id', 'unknown'),
            requirements=requirements,
            overall_scope=scope,
            technical_stack=tech_stack,
            quality_requirements=quality_reqs,
            constraints=constraints,
            assumptions=assumptions,
            risks=risks
        )
    
    async def _extract_requirements(self, text: str) -> List[ParsedRequirement]:
        """Extract individual requirements from text."""
        requirements = []
        
        # Split text into logical sections
        sections = self._split_into_sections(text)
        
        for section_idx, section in enumerate(sections):
            # Extract requirements from each section
            section_requirements = self._parse_section_requirements(section, section_idx)
            requirements.extend(section_requirements)
        
        return requirements
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections for requirement extraction."""
        
        # Common section delimiters
        section_patterns = [
            r'\n#{1,4}\s+(.+)',  # Markdown headers
            r'\n\*\*(.+)\*\*',   # Bold text
            r'\n(\d+\.\s+.+)',   # Numbered lists
            r'\n[-*]\s+(.+)',    # Bullet points
            r'\n[A-Z][^.!?]*[.!?]'  # Sentences starting with capital letters
        ]
        
        sections = []
        current_section = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            is_new_section = False
            for pattern in section_patterns:
                if re.match(pattern, f'\n{line}'):
                    is_new_section = True
                    break
            
            if is_new_section and current_section:
                sections.append(current_section.strip())
                current_section = line
            else:
                current_section += f" {line}"
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections if sections else [text]
    
    def _parse_section_requirements(self, section: str, section_idx: int) -> List[ParsedRequirement]:
        """Parse requirements from a text section."""
        requirements = []
        
        for category, patterns in self.requirement_patterns.items():
            for pattern_idx, pattern in enumerate(patterns):
                matches = re.finditer(pattern, section, re.IGNORECASE | re.MULTILINE)
                
                for match_idx, match in enumerate(matches):
                    requirement_text = match.group(0)
                    
                    # Determine priority
                    priority = self._determine_priority(requirement_text)
                    
                    # Extract acceptance criteria
                    acceptance_criteria = self._extract_acceptance_criteria(section, requirement_text)
                    
                    # Extract dependencies
                    dependencies = self._extract_dependencies(requirement_text)
                    
                    # Extract technical constraints
                    tech_constraints = self._extract_technical_constraints(requirement_text)
                    
                    # Estimate complexity
                    complexity = self._estimate_requirement_complexity(requirement_text)
                    
                    # Calculate confidence
                    confidence = self._calculate_parsing_confidence(requirement_text, match)
                    
                    requirement = ParsedRequirement(
                        requirement_id=f"req_{section_idx}_{pattern_idx}_{match_idx}",
                        category=category,
                        priority=priority,
                        description=requirement_text.strip(),
                        acceptance_criteria=acceptance_criteria,
                        dependencies=dependencies,
                        technical_constraints=tech_constraints,
                        estimated_complexity=complexity,
                        confidence_score=confidence
                    )
                    
                    requirements.append(requirement)
        
        return requirements
    
    def _determine_priority(self, text: str) -> str:
        """Determine requirement priority from text."""
        text_lower = text.lower()
        
        for priority, indicators in self.priority_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return priority
        
        return 'should'  # Default priority
    
    def _extract_acceptance_criteria(self, full_section: str, requirement_text: str) -> List[str]:
        """Extract acceptance criteria related to a requirement."""
        criteria = []
        
        # First, extract from existing acceptance criteria patterns
        for pattern in self.acceptance_patterns:
            matches = re.finditer(pattern, full_section, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                criterion = match.group(1) if match.lastindex and len(match.groups()) > 0 else match.group(0)
                # Split on newlines and clean up bullet points
                lines = criterion.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('acceptance criteria', 'ac:', 'definition of done', 'dod:')):
                        # Remove bullet points and dashes
                        clean_line = re.sub(r'^[-*•]\s*', '', line)
                        if clean_line:
                            criteria.append(clean_line)
        
        # Look for Given-When-Then patterns - handle both inline and multiline
        lines = full_section.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for Given-When-Then starting on this line
            if line.lower().startswith('given '):
                given_part = line[6:].strip()  # Remove "Given "
                when_part = ""
                then_part = ""
                
                # Look for When on next lines
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.lower().startswith('when '):
                        when_part = next_line[5:].strip()  # Remove "When "
                        break
                    elif next_line and not next_line.lower().startswith('given '):
                        # Continue Given part if no When found yet
                        given_part += " " + next_line
                    j += 1
                
                # Look for Then after When
                if when_part:
                    k = j + 1
                    while k < len(lines):
                        next_line = lines[k].strip()
                        if next_line.lower().startswith('then '):
                            then_part = next_line[5:].strip()  # Remove "Then "
                            break
                        elif next_line and not next_line.lower().startswith(('given ', 'when ')):
                            # Continue When part if no Then found yet
                            when_part += " " + next_line
                        k += 1
                
                # If we found all three parts, create the criterion
                if given_part and when_part and then_part:
                    gherkin_criterion = f"Given {given_part}, When {when_part}, Then {then_part}"
                    criteria.append(gherkin_criterion)
                    i = k  # Skip to after Then
                else:
                    i += 1
            else:
                i += 1
        
        # Also try inline Given-When-Then pattern as fallback
        gherkin_pattern = r'given\s+(.+?)\s+when\s+(.+?)\s+then\s+(.+?)(?=\n(?:\s*\n|\s*[A-Z])|$)'
        gherkin_matches = re.finditer(gherkin_pattern, full_section, re.IGNORECASE | re.DOTALL)
        
        for match in gherkin_matches:
            given, when, then = match.groups()
            gherkin_criterion = f"Given {given.strip()}, When {when.strip()}, Then {then.strip()}"
            # Only add if we haven't already found this pattern
            if not any(gherkin_criterion in existing for existing in criteria):
                criteria.append(gherkin_criterion)
        
        return criteria
            
    def _extract_dependencies(self, text: str) -> List[str]:
        """Extract dependencies from requirement text."""
        dependencies = []
        
        dependency_patterns = [
            r'depends on (.+)',
            r'requires (.+)',
            r'needs (.+) (to be|before)',
            r'after (.+) is (complete|done|implemented)',
            r'blocked by (.+)'
        ]
        
        for pattern in dependency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dependency = match.group(1).strip()
                dependencies.append(dependency)
        
        return dependencies
    
    def _extract_technical_constraints(self, text: str) -> List[str]:
        """Extract technical constraints from requirement text."""
        constraints = []
        
        for pattern in self.constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.group(0).strip()
                constraints.append(constraint)
        
        return constraints
    
    def _estimate_requirement_complexity(self, text: str) -> float:
        """Estimate complexity of a single requirement."""
        complexity = 0.3  # Base complexity
        
        # Length indicates complexity
        word_count = len(text.split())
        if word_count > 20:
            complexity += 0.2
        elif word_count > 10:
            complexity += 0.1
        
        # Technical terms increase complexity
        technical_terms = ['algorithm', 'optimization', 'integration', 'security', 'performance']
        if any(term in text.lower() for term in technical_terms):
            complexity += 0.3
        
        # Multiple components
        if len(re.findall(r'\band\b|\bor\b', text, re.IGNORECASE)) > 2:
            complexity += 0.2
        
        # Conditional logic
        if any(word in text.lower() for word in ['if', 'when', 'unless', 'except']):
            complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _calculate_parsing_confidence(self, text: str, match: re.Match) -> float:
        """Calculate confidence in requirement parsing."""
        confidence = 0.7  # Base confidence
        
        # Strong pattern match
        if match.group(0) == text.strip():
            confidence += 0.2
        
        # Clear structure
        if any(indicator in text.lower() for indicator in ['must', 'shall', 'should']):
            confidence += 0.1
        
        # Complete sentence
        if text.strip().endswith('.'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _determine_scope(self, text: str, requirements: List[ParsedRequirement]) -> str:
        """Determine overall scope of the task."""
        
        # Count requirements by category
        functional_count = len([r for r in requirements if r.category == 'functional'])
        non_functional_count = len([r for r in requirements if r.category == 'non_functional'])
        technical_count = len([r for r in requirements if r.category == 'technical'])
        
        total_requirements = len(requirements)
        
        # Analyze text indicators
        scope_indicators = {
            'epic': ['epic', 'major', 'complete rewrite', 'full implementation'],
            'large': ['large', 'significant', 'comprehensive', 'multiple'],
            'medium': ['medium', 'moderate', 'standard', 'typical'],
            'small': ['small', 'minor', 'simple', 'quick']
        }
        
        text_lower = text.lower()
        for scope, indicators in scope_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return scope
        
        # Determine by requirement count and complexity
        avg_complexity = sum(r.estimated_complexity for r in requirements) / max(total_requirements, 1)
        
        if total_requirements > 10 or avg_complexity > 0.8:
            return 'epic'
        elif total_requirements > 5 or avg_complexity > 0.6:
            return 'large'
        elif total_requirements > 2 or avg_complexity > 0.4:
            return 'medium'
        else:
            return 'small'
    
    def _extract_technical_stack(self, text: str, labels: List[str]) -> List[str]:
        """Extract mentioned technologies and technical stack."""
        tech_stack = []
        
        # Technology patterns
        tech_patterns = {
            'languages': r'\b(python|javascript|typescript|java|cpp|c\+\+|go|rust|ruby|php|kotlin|swift)\b',
            'frameworks': r'\b(react|angular|vue|django|flask|spring|express|rails|laravel|fastapi)\b',
            'databases': r'\b(mysql|postgresql|mongodb|redis|elasticsearch|sqlite|oracle)\b',
            'cloud': r'\b(aws|azure|gcp|google cloud|amazon|microsoft|docker|kubernetes)\b',
            'tools': r'\b(git|jenkins|gitlab|github|jira|confluence|slack)\b'
        }
        
        text_lower = text.lower()
        for category, pattern in tech_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            tech_stack.extend(matches)
        
        # Also check labels
        for label in labels:
            label_lower = label.lower()
            for pattern in tech_patterns.values():
                matches = re.findall(pattern, label_lower, re.IGNORECASE)
                tech_stack.extend(matches)
        
        return list(set(tech_stack))  # Remove duplicates
    
    def _extract_quality_requirements(self, text: str) -> Dict[str, str]:
        """Extract quality and non-functional requirements."""
        quality_reqs = {}
        
        quality_patterns = {
            'performance': r'(performance|response time|latency|throughput)[:.]?\s*(.+?)(?:\n|$)',
            'security': r'(security|authentication|authorization|encryption)[:.]?\s*(.+?)(?:\n|$)',
            'scalability': r'(scalability|scale|concurrent users)[:.]?\s*(.+?)(?:\n|$)',
            'availability': r'(availability|uptime|downtime)[:.]?\s*(.+?)(?:\n|$)',
            'usability': r'(usability|user experience|ease of use)[:.]?\s*(.+?)(?:\n|$)',
            'reliability': r'(reliability|stability|error rate)[:.]?\s*(.+?)(?:\n|$)'
        }
        
        for quality_type, pattern in quality_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                requirement = match.group(2).strip() if match.lastindex >= 2 else match.group(0)
                quality_reqs[quality_type] = requirement
        
        return quality_reqs
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract project constraints."""
        constraints = []
        
        for pattern in self.constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                constraint = match.group(0).strip()
                constraints.append(constraint)
        
        # Timeline constraints
        timeline_pattern = r'(deadline|due date|delivery|complete by)[:.]?\s*(.+?)(?:\n|$)'
        timeline_matches = re.finditer(timeline_pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in timeline_matches:
            constraints.append(f"Timeline: {match.group(0).strip()}")
        
        return constraints
    
    def _extract_assumptions(self, text: str) -> List[str]:
        """Extract project assumptions."""
        assumptions = []
        
        assumption_patterns = [
            r'assume\s+(.+?)(?:\n|$)',
            r'assumption[:.]?\s*(.+?)(?:\n|$)',
            r'assuming\s+(.+?)(?:\n|$)',
            r'we expect\s+(.+?)(?:\n|$)'
        ]
        
        for pattern in assumption_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                assumption = match.group(1).strip()
                assumptions.append(assumption)
        
        return assumptions
    
    def _identify_requirement_risks(self, text: str, requirements: List[ParsedRequirement]) -> List[str]:
        """Identify potential risks from requirements analysis."""
        risks = []
        
        # Complexity risks
        high_complexity_reqs = [r for r in requirements if r.estimated_complexity > 0.7]
        if len(high_complexity_reqs) > 3:
            risks.append("Multiple high-complexity requirements may impact timeline")
        
        # Dependency risks
        deps_count = sum(len(r.dependencies) for r in requirements)
        if deps_count > 5:
            risks.append("High number of dependencies may cause delays")
        
        # Scope risks
        if len(requirements) > 10:
            risks.append("Large number of requirements - scope creep risk")
        
        # Technical risks
        text_lower = text.lower()
        risk_indicators = {
            'integration': 'External integration complexity',
            'migration': 'Data migration risks',
            'performance': 'Performance optimization challenges',
            'security': 'Security implementation complexity',
            'legacy': 'Legacy system integration challenges'
        }
        
        for indicator, risk_desc in risk_indicators.items():
            if indicator in text_lower:
                risks.append(risk_desc)
        
        # Ambiguity risks
        vague_requirements = [r for r in requirements if r.confidence_score < 0.6]
        if len(vague_requirements) > 2:
            risks.append("Ambiguous requirements may need clarification")
        
        return risks

    def validate_requirements(self, requirements: TaskRequirements) -> Dict[str, List[str]]:
        """Validate requirements for completeness and consistency."""
        validation_results = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for missing acceptance criteria
        reqs_without_ac = [r for r in requirements.requirements if not r.acceptance_criteria]
        if len(reqs_without_ac) > len(requirements.requirements) * 0.5:
            validation_results['warnings'].append("Many requirements lack acceptance criteria")
        
        # Check for conflicting requirements
        must_requirements = [r for r in requirements.requirements if r.priority == 'must']
        if len(must_requirements) > 10:
            validation_results['warnings'].append("Too many 'must have' requirements - consider prioritization")
        
        # Check scope consistency
        if requirements.overall_scope == 'small' and len(requirements.requirements) > 5:
            validation_results['errors'].append("Scope marked as 'small' but has many requirements")
        
        # Check for low confidence requirements
        low_confidence_reqs = [r for r in requirements.requirements if r.confidence_score < 0.6]
        if len(low_confidence_reqs) > 0:
            validation_results['warnings'].append(f"{len(low_confidence_reqs)} requirements have low confidence scores")
        
        # Technical stack consistency
        mentioned_techs = set()
        for req in requirements.requirements:
            if req.category == 'technical':
                mentioned_techs.update(req.technical_constraints)
        
        if len(mentioned_techs) > 5:
            validation_results['warnings'].append("Multiple technologies mentioned - complexity risk")
        
        return validation_results