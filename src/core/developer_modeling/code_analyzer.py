import ast
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

from config import settings

@dataclass
class CodeMetrics:
    complexity_score: float
    language_distribution: Dict[str, float]
    technical_concepts: List[str]
    semantic_embedding: np.ndarray
    function_count: int
    class_count: int
    import_complexity: float

class CodeAnalyzer:
    """Analyzes code commits for developer skill extraction."""
    
    def __init__(self):
        # Load pre-trained models
        self.tokenizer = AutoTokenizer.from_pretrained(settings.CODEBERT_MODEL)
        self.codebert_model = AutoModel.from_pretrained(settings.CODEBERT_MODEL)
        self.sentence_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        
        # Language detection patterns
        self.language_patterns = {
            'python': [r'\.py$', r'def\s+\w+', r'import\s+\w+', r'class\s+\w+'],
            'javascript': [r'\.js$', r'function\s+\w+', r'const\s+\w+', r'=>\s*{'],
            'typescript': [r'\.ts$', r'interface\s+\w+', r'type\s+\w+', r':\s*\w+'],
            'java': [r'\.java$', r'public\s+class', r'private\s+\w+', r'@\w+'],
            'cpp': [r'\.cpp$', r'\.hpp$', r'#include', r'std::'],
            'go': [r'\.go$', r'func\s+\w+', r'package\s+\w+', r'import\s+"'],
            'rust': [r'\.rs$', r'fn\s+\w+', r'let\s+mut', r'impl\s+\w+'],
        }
        
        # Technical concept patterns
        self.concept_patterns = {
            'database': [r'sql', r'query', r'select', r'insert', r'update', r'delete', 
                        r'database', r'table', r'schema', r'migration'],
            'api': [r'api', r'rest', r'graphql', r'endpoint', r'request', r'response',
                   r'http', r'get', r'post', r'put', r'delete'],
            'frontend': [r'react', r'vue', r'angular', r'component', r'jsx', r'tsx',
                        r'css', r'html', r'dom', r'event'],
            'backend': [r'server', r'middleware', r'auth', r'session', r'cookie',
                       r'route', r'handler', r'service'],
            'testing': [r'test', r'spec', r'mock', r'assert', r'expect', r'jest',
                       r'pytest', r'unittest'],
            'devops': [r'docker', r'kubernetes', r'ci', r'cd', r'pipeline', r'deploy',
                      r'aws', r'azure', r'gcp'],
            'ml': [r'machine learning', r'neural', r'model', r'train', r'predict',
                  r'tensorflow', r'pytorch', r'sklearn'],
            'security': [r'auth', r'jwt', r'oauth', r'encrypt', r'hash', r'security',
                        r'vulnerability', r'sanitize']
        }
    
    def analyze_commit(self, files_changed: List[Dict], commit_info: Dict) -> CodeMetrics:
        """Analyze a single commit for code complexity and technical concepts."""
        
        all_code = ""
        language_lines = defaultdict(int)
        total_lines = 0
        function_count = 0
        class_count = 0
        technical_concepts = []
        
        for file_info in files_changed:
            filename = file_info.get('filename', '')
            patch = file_info.get('patch', '')
            additions = file_info.get('additions', 0)
            
            # Detect language
            language = self._detect_language(filename, patch)
            if language:
                language_lines[language] += additions
                total_lines += additions
            
            # Extract added code lines
            added_lines = self._extract_added_lines(patch)
            all_code += "\n".join(added_lines) + "\n"
            
            # Count functions and classes
            function_count += self._count_functions(added_lines, language)
            class_count += self._count_classes(added_lines, language)
            
            # Extract technical concepts
            concepts = self._extract_technical_concepts(added_lines)
            technical_concepts.extend(concepts)
        
        # Calculate language distribution
        language_distribution = {}
        if total_lines > 0:
            for lang, lines in language_lines.items():
                language_distribution[lang] = lines / total_lines
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(all_code, function_count, class_count)
        
        # Generate semantic embedding
        semantic_embedding = self._generate_code_embedding(all_code)
        
        # Calculate import complexity
        import_complexity = self._calculate_import_complexity(all_code)
        
        return CodeMetrics(
            complexity_score=complexity_score,
            language_distribution=language_distribution,
            technical_concepts=list(set(technical_concepts)),
            semantic_embedding=semantic_embedding,
            function_count=function_count,
            class_count=class_count,
            import_complexity=import_complexity
        )
    
    def _detect_language(self, filename: str, content: str) -> Optional[str]:
        """Detect programming language from filename and content."""
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    score += 2  # Filename match is strong indicator
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    score += 1
            
            if score >= 2:  # Threshold for language detection
                return language
        
        return None
    
    def _extract_added_lines(self, patch: str) -> List[str]:
        """Extract only the added lines from a git patch."""
        if not patch:
            return []
        
        added_lines = []
        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                # Remove the '+' prefix and add to list
                added_lines.append(line[1:])
        
        return added_lines
    
    def _count_functions(self, lines: List[str], language: Optional[str]) -> int:
        """Count function definitions in code lines."""
        if not language:
            return 0
        
        function_patterns = {
            'python': r'^\s*def\s+\w+',
            'javascript': r'^\s*function\s+\w+|^\s*\w+\s*=\s*\(',
            'typescript': r'^\s*function\s+\w+|^\s*\w+\s*=\s*\(',
            'java': r'^\s*(public|private|protected).*\w+\s*\(',
            'cpp': r'^\s*\w+.*\w+\s*\([^;]*\)\s*{',
            'go': r'^\s*func\s+\w+',
            'rust': r'^\s*fn\s+\w+',
        }
        
        pattern = function_patterns.get(language, r'^\s*\w+.*\(')
        count = 0
        
        for line in lines:
            if re.search(pattern, line):
                count += 1
        
        return count
    
    def _count_classes(self, lines: List[str], language: Optional[str]) -> int:
        """Count class definitions in code lines."""
        if not language:
            return 0
        
        class_patterns = {
            'python': r'^\s*class\s+\w+',
            'javascript': r'^\s*class\s+\w+',
            'typescript': r'^\s*class\s+\w+|^\s*interface\s+\w+',
            'java': r'^\s*(public|private)?\s*class\s+\w+',
            'cpp': r'^\s*class\s+\w+',
            'go': r'^\s*type\s+\w+\s+struct',
            'rust': r'^\s*struct\s+\w+|^\s*impl\s+\w+',
        }
        
        pattern = class_patterns.get(language, r'^\s*class\s+\w+')
        count = 0
        
        for line in lines:
            if re.search(pattern, line):
                count += 1
        
        return count
    
    def _extract_technical_concepts(self, lines: List[str]) -> List[str]:
        """Extract technical concepts from code lines."""
        concepts = []
        content = " ".join(lines).lower()
        
        for domain, keywords in self.concept_patterns.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}\b', content):
                    concepts.append(domain)
                    break  # Only add domain once
        
        return concepts
    
    def _calculate_complexity(self, code: str, function_count: int, class_count: int) -> float:
        """Calculate code complexity score."""
        if not code.strip():
            return 0.0
        
        # Basic complexity indicators
        lines = [line for line in code.split('\n') if line.strip()]
        line_count = len(lines)
        
        # Control flow complexity
        control_keywords = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'try', 'catch']
        control_count = sum(len(re.findall(rf'\b{keyword}\b', code.lower())) for keyword in control_keywords)
        
        # Nesting depth (approximate)
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # Assuming 4-space indentation
        
        # Normalize complexity score (0-1 scale)
        complexity = (
            min(line_count / 100, 1.0) * 0.3 +  # Line complexity
            min(control_count / 20, 1.0) * 0.3 +  # Control flow complexity
            min(max_indent / 10, 1.0) * 0.2 +  # Nesting complexity
            min(function_count / 10, 1.0) * 0.1 +  # Function complexity
            min(class_count / 5, 1.0) * 0.1  # Class complexity
        )
        
        return min(complexity, 1.0)
    
    def _generate_code_embedding(self, code: str) -> np.ndarray:
        """Generate semantic embedding for code using CodeBERT."""
        if not code.strip():
            return np.zeros(settings.SKILL_VECTOR_DIM)
        
        try:
            # Truncate code to model's max length
            tokens = self.tokenizer.tokenize(code)
            if len(tokens) > 510:  # Leave room for special tokens
                tokens = tokens[:510]
                code = self.tokenizer.convert_tokens_to_string(tokens)
            
            # Encode with CodeBERT
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.codebert_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[0][0].numpy()
            
            return embedding
        
        except Exception as e:
            print(f"Error generating code embedding: {e}")
            return np.zeros(settings.SKILL_VECTOR_DIM)
    
    def _calculate_import_complexity(self, code: str) -> float:
        """Calculate complexity based on imports and dependencies."""
        import_patterns = [
            r'import\s+[\w\.]+',
            r'from\s+[\w\.]+\s+import',
            r'#include\s*<[\w\.]+>',
            r'require\s*\([\'"][\w\.\/]+[\'"]\)',
            r'use\s+[\w::]+',
        ]
        
        import_count = 0
        for pattern in import_patterns:
            import_count += len(re.findall(pattern, code, re.IGNORECASE))
        
        # Normalize to 0-1 scale
        return min(import_count / 20, 1.0)
    
    def extract_developer_skills(self, code_analyses: List[CodeMetrics], 
                                time_window_days: int = 90) -> Dict:
        """Extract comprehensive skill profile from multiple code analyses."""
        
        if not code_analyses:
            return {
                'programming_languages': {},
                'domain_expertise': {},
                'complexity_preference': 0.0,
                'technical_breadth': 0.0,
                'code_quality_indicator': 0.0
            }
        
        # Aggregate language skills
        language_skills = defaultdict(list)
        all_concepts = []
        complexity_scores = []
        
        for analysis in code_analyses:
            # Language proficiency
            for lang, ratio in analysis.language_distribution.items():
                language_skills[lang].append(ratio)
            
            # Technical concepts
            all_concepts.extend(analysis.technical_concepts)
            
            # Complexity handling
            complexity_scores.append(analysis.complexity_score)
        
        # Calculate language proficiency scores
        programming_languages = {}
        for lang, ratios in language_skills.items():
            programming_languages[lang] = {
                'proficiency': np.mean(ratios),
                'consistency': 1.0 - np.std(ratios) if len(ratios) > 1 else 1.0,
                'recent_usage': sum(ratios[-10:]) / min(len(ratios), 10)  # Recent activity
            }
        
        # Calculate domain expertise
        concept_counts = Counter(all_concepts)
        total_concepts = len(all_concepts)
        domain_expertise = {}
        
        for domain, count in concept_counts.items():
            domain_expertise[domain] = {
                'frequency': count / total_concepts if total_concepts > 0 else 0,
                'expertise_level': min(count / 10, 1.0)  # Normalize to 0-1
            }
        
        # Calculate metrics
        complexity_preference = np.mean(complexity_scores) if complexity_scores else 0.0
        technical_breadth = len(set(all_concepts)) / 20  # Normalize
        code_quality_indicator = self._estimate_code_quality(code_analyses)
        
        return {
            'programming_languages': programming_languages,
            'domain_expertise': domain_expertise,
            'complexity_preference': complexity_preference,
            'technical_breadth': min(technical_breadth, 1.0),
            'code_quality_indicator': code_quality_indicator
        }
    
    def _estimate_code_quality(self, analyses: List[CodeMetrics]) -> float:
        """Estimate code quality based on various indicators."""
        if not analyses:
            return 0.0
        
        quality_indicators = []
        
        for analysis in analyses:
            # Function/class ratio indicates good structure
            total_constructs = analysis.function_count + analysis.class_count
            if total_constructs > 0:
                structure_score = min(total_constructs / 5, 1.0)
            else:
                structure_score = 0.0
            
            # Moderate complexity is often better than very high or very low
            complexity_quality = 1.0 - abs(analysis.complexity_score - 0.5)
            
            # Import complexity indicates thoughtful dependency management
            import_quality = min(analysis.import_complexity, 0.8)  # Cap at 0.8
            
            quality = (structure_score * 0.4 + complexity_quality * 0.4 + import_quality * 0.2)
            quality_indicators.append(quality)
        
        return np.mean(quality_indicators)