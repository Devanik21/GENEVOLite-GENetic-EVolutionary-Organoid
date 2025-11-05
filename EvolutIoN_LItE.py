"""
GENEVO: Advanced Neuroevolutionary System for AGI
A scientifically rigorous implementation of genetic neural architecture evolution

Mathematical Foundation:

1. Phenotypic Development & Fitness Vector (Performance, Cost, Robustness):
   V(G, E(t_evo)) = E_{d ~ P(D|E)} [ (1 - L_task(P', d)), C(P')⁻¹, R(P', d, η_pert) ]
   where P' = argmin_{P*} ∫ (∇_{P*} L_task) dτ  (Lifetime Learning)
   and   P = φ(G, E, η_dev) (Development)

2. Population Evolutionary Dynamics (Non-local Fokker-Planck Equation):
   ∂ρ(G,t)/∂t = ∇_G ⋅ [D(G)∇_G ρ] - ∇_G ⋅ [ρ M(G) ∫ K(G,G') s(V(G),V(G')) ρ(G') dG']
   This describes the evolution of population density ρ(G,t) under mutation (diffusion),
   and multi-objective selection (non-local drift based on Pareto dominance).

3. Robust Multi-Objective Goal (Minimax Regret over Environment Space):
   Find G* ⊂ Γ  s.t.  G* = argmin_{G' ⊂ Γ} sup_{E ∈ E} d_H(V(G', E), P*(E))
   The goal is to find a portfolio of genotypes G* that minimizes the worst-case
   Hausdorff distance (d_H) to the true Pareto Front (P*) over all possible environments (E).

This system implements:
1. Indirect encoding via developmental programs
2. Compositional evolution with hierarchical modules  
3. Multi-objective optimization (accuracy, efficiency, complexity)
4. Coevolution with dynamic fitness landscapes
5. Baldwin effect modeling
6. Speciation and Niche Competition
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
import random
import time
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
from tinydb import TinyDB, Query
from collections import Counter
import json

# ==================== THEORETICAL FOUNDATIONS ====================

class EvolutionaryTheory:
    """Mathematical framework for neuroevolution"""
    
    @staticmethod
    def fisher_information(population: List, fitness: np.ndarray) -> float:
        """Fisher information measures evolutionary potential"""
        if len(fitness) < 2:
            return 0.0
        variance = np.var(fitness)
        return 1.0 / (variance + 1e-8)
    
    @staticmethod
    def genetic_diversity(population: List) -> float:
        """Shannon entropy of genotypic diversity"""
        if not population:
            return 0.0
        sizes = [sum(m.size for m in ind.modules) for ind in population]
        hist, _ = np.histogram(sizes, bins=10)
        probs = hist / (hist.sum() + 1e-8)
        return entropy(probs + 1e-8)
    
    @staticmethod
    def selection_differential(fitness: np.ndarray, selected_idx: np.ndarray) -> float:
        """Measure of selection pressure"""
        if len(fitness) == 0:
            return 0.0
        mean_all = np.mean(fitness)
        mean_selected = np.mean(fitness[selected_idx])
        return mean_selected - mean_all
    
    @staticmethod
    def heritability(parent_fitness: np.ndarray, offspring_fitness: np.ndarray) -> float:
        """Estimate narrow-sense heritability h²"""
        if len(parent_fitness) < 2 or len(offspring_fitness) < 2:
            return 0.0
        min_len = min(len(parent_fitness), len(offspring_fitness))
        if min_len < 2:
            return 0.0
        try:
            corr, _ = pearsonr(parent_fitness[:min_len], offspring_fitness[:min_len])
            return max(0, min(1, corr))
        except:
            return 0.0

# ==================== ADVANCED GENOTYPE ====================

@dataclass
class DevelopmentalGene:
    """Encodes developmental rules for phenotype construction"""
    rule_type: str  # 'proliferation', 'differentiation', 'migration', 'pruning'
    trigger_condition: str  # When this rule activates
    parameters: Dict[str, float]
    
@dataclass
class ModuleGene:
    """Enhanced module gene with biological properties"""
    id: str
    module_type: str
    size: int
    activation: str  # 'relu', 'gelu', 'silu', 'swish'
    normalization: str  # 'batch', 'layer', 'instance', 'none'
    dropout_rate: float
    learning_rate_mult: float  # Local learning rate multiplier
    plasticity: float  # Capacity for lifetime learning
    color: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D coordinates
    
@dataclass
class ConnectionGene:
    """Enhanced connection with synaptic properties"""
    source: str
    target: str
    weight: float
    connection_type: str  # 'excitatory', 'inhibitory', 'modulatory'
    delay: float  # Signal propagation delay
    plasticity_rule: str  # 'hebbian', 'anti-hebbian', 'stdp', 'static'
    
@dataclass
class Genotype:
    """Complete genetic encoding with developmental program"""
    modules: List[ModuleGene]
    connections: List[ConnectionGene]
    developmental_rules: List[DevelopmentalGene] = field(default_factory=list)
    meta_parameters: Dict[str, float] = field(default_factory=dict)
    
    epigenetic_markers: Dict[str, float] = field(default_factory=dict)
    # Evolutionary metrics
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    lineage_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    
    # Performance characteristics
    accuracy: float = 0.0
    efficiency: float = 0.0  # Compute cost normalized
    complexity: float = 0.0  # Architectural complexity
    robustness: float = 0.0  # Performance under perturbation
    
    form_id: int = 1
    
    def __post_init__(self):
        if not self.lineage_id:
            self.lineage_id = f"L{random.randint(0, 999999):06d}"
        if 'learning_rate' not in self.meta_parameters:
            self.meta_parameters = {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'temperature': 1.0,
                'exploration_noise': 0.1
            }
    
    def copy(self):
        """Deep copy with new lineage"""
        new_genotype = Genotype(
            modules=[ModuleGene(
                m.id, m.module_type, m.size, m.activation, m.normalization,
                m.dropout_rate, m.learning_rate_mult, m.plasticity, m.color, m.position
            ) for m in self.modules],
            connections=[ConnectionGene(
                c.source, c.target, c.weight, c.connection_type, c.delay, c.plasticity_rule
            ) for c in self.connections],
            developmental_rules=[DevelopmentalGene(
                d.rule_type, d.trigger_condition, d.parameters.copy()
            ) for d in self.developmental_rules],
            meta_parameters=self.meta_parameters.copy(),
            epigenetic_markers={k: v * 0.5 for k, v in self.epigenetic_markers.items()}, # Imperfect inheritance
            fitness=self.fitness,
            age=0,
            generation=self.generation,
            parent_ids=[self.lineage_id],
            form_id=self.form_id
        )
        return new_genotype
    
    def compute_complexity(self) -> float:
        """Kolmogorov complexity approximation"""
        param_count = sum(m.size for m in self.modules)
        connection_count = len(self.connections)
        module_diversity = len(set(m.module_type for m in self.modules))
        
        # Normalized complexity score
        c_params = np.log(1 + param_count) / 20
        c_connections = len(self.connections) / (len(self.modules) ** 2 + 1)
        c_diversity = module_diversity / 10
        
        return (c_params + c_connections + c_diversity) / 3

def genotype_to_dict(g: Genotype) -> Dict:
    """Serializes a Genotype object to a dictionary."""
    return asdict(g)

def dict_to_genotype(d: Dict) -> Genotype:
    """Deserializes a dictionary back into a Genotype object."""
    # Reconstruct nested dataclasses
    d['modules'] = [ModuleGene(**m) for m in d.get('modules', [])]
    d['connections'] = [ConnectionGene(**c) for c in d.get('connections', [])]
    d['developmental_rules'] = [DevelopmentalGene(**dr) for dr in d.get('developmental_rules', [])]
    
    # The Genotype dataclass can now be instantiated with the dictionary
    return Genotype(**d)

def genomic_distance(g1: Genotype, g2: Genotype, c1=1.0, c3=0.5) -> float:
    """
    Computes genomic distance between two genotypes (NEAT-inspired).
    Distance is a weighted sum of differences in connections and module properties.
    c1: weight for disjoint/excess connections
    c3: weight for average attribute difference of matching modules (e.g., size)
    """
    # Incompatible forms are in different species
    if g1.form_id != g2.form_id:
        return float('inf')

    # Compare connections
    g1_conns = {(c.source, c.target) for c in g1.connections}
    g2_conns = {(c.source, c.target) for c in g2.connections}
    
    disjoint_conns = len(g1_conns.symmetric_difference(g2_conns))
    
    # Compare modules
    g1_modules = {m.id: m for m in g1.modules}
    g2_modules = {m.id: m for m in g2.modules}
    
    matching_modules = 0
    size_diff = 0.0
    plasticity_diff = 0.0
    
    all_module_ids = set(g1_modules.keys()) | set(g2_modules.keys())
    
    for mid in all_module_ids:
        if mid in g1_modules and mid in g2_modules:
            matching_modules += 1
            size_diff += abs(g1_modules[mid].size - g2_modules[mid].size)
            plasticity_diff += abs(g1_modules[mid].plasticity - g2_modules[mid].plasticity)

    avg_size_diff = (size_diff / matching_modules) if matching_modules > 0 else 0
    avg_plasticity_diff = (plasticity_diff / matching_modules) if matching_modules > 0 else 0

    N = max(1, len(g1.connections), len(g2.connections))
    distance = (c1 * disjoint_conns / N) + (c3 * (avg_size_diff / 100 + avg_plasticity_diff))
    
    return distance

def is_viable(genotype: Genotype) -> bool:
    """
    Checks if a genotype is structurally viable, meaning it has a path
    from an input node to an output node. This prevents non-functional
    architectures from entering the population.
    """
    if not genotype.modules or not genotype.connections:
        return False

    G = nx.DiGraph()
    module_ids = {m.id for m in genotype.modules}
    
    for conn in genotype.connections:
        # Ensure connections are between existing modules
        if conn.source in module_ids and conn.target in module_ids:
            G.add_edge(conn.source, conn.target)

    if G.number_of_nodes() < 2: # Need at least two nodes to form a path
        return False

    # Identify input nodes (in-degree == 0) and output nodes (out-degree == 0)
    input_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    output_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]

    # If there are no clear input/output nodes (e.g., a cycle), use heuristics
    if not input_nodes:
        potential_inputs = [m.id for m in genotype.modules if 'input' in m.id or 'embed' in m.id or 'V1' in m.id]
        input_nodes = [node for node in potential_inputs if node in G.nodes]

    if not output_nodes:
        potential_outputs = [m.id for m in genotype.modules if 'output' in m.id or 'PFC' in m.id]
        output_nodes = [node for node in potential_outputs if node in G.nodes]

    if not input_nodes or not output_nodes:
        return False # Cannot determine a start or end point

    # Check for a path from any input to any output
    for start_node in input_nodes:
        for end_node in output_nodes:
            if start_node in G and end_node in G and nx.has_path(G, start_node, end_node):
                return True

    return False

# ==================== ADVANCED INITIALIZATION ====================

def initialize_genotype(form_id: int, complexity_level: str = 'medium') -> Genotype:
    """Initialize genotype with scientifically grounded architectures"""
    
    complexity_scales = {
        'minimal': (32, 128, 0.3),
        'medium': (64, 256, 0.5),
        'high': (128, 512, 0.8)
    }
    
    base_size, max_size, connection_density = complexity_scales[complexity_level]
    
    forms = {
        1: {  # Convolutional Cascade (Visual Processing)
            'name': 'Hierarchical Convolutional',
            'modules': [
                ModuleGene('V1', 'conv', base_size, 'relu', 'batch', 0.1, 1.0, 0.3, '#FF6B6B', (0, 0, 0)),
                ModuleGene('V2', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.4, '#FD79A8', (1, 0, 0)),
                ModuleGene('V4', 'conv', base_size*3, 'gelu', 'layer', 0.2, 0.8, 0.5, '#FDCB6E', (2, 0, 0)),
                ModuleGene('IT', 'attention', base_size*4, 'gelu', 'layer', 0.25, 0.6, 0.6, '#00B894', (3, 0, 0)),
                ModuleGene('PFC', 'mlp', max_size, 'swish', 'layer', 0.3, 0.5, 0.7, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'hierarchical',
            'inductive_bias': 'spatial_locality'
        },
        2: {  # Attention Network (Transformer-like)
            'name': 'Multi-Head Attention System',
            'modules': [
                ModuleGene('embed', 'mlp', base_size*2, 'gelu', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('attn_1', 'attention', base_size*4, 'gelu', 'layer', 0.1, 1.0, 0.6, '#FECA57', (1, 1, 0)),
                ModuleGene('attn_2', 'attention', base_size*4, 'gelu', 'layer', 0.15, 0.9, 0.6, '#48DBFB', (2, 1, 0)),
                ModuleGene('attn_3', 'attention', base_size*4, 'gelu', 'layer', 0.2, 0.8, 0.6, '#A29BFE', (3, 1, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'residual_attention',
            'inductive_bias': 'long_range_dependencies'
        },
        3: {  # Recurrent Memory Network
            'name': 'Dynamical Recurrent System',
            'modules': [
                ModuleGene('input_gate', 'mlp', base_size, 'sigmoid', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('lstm_1', 'recurrent', base_size*3, 'tanh', 'layer', 0.2, 1.0, 0.8, '#A29BFE', (1, 0, 1)),
                ModuleGene('lstm_2', 'recurrent', base_size*3, 'tanh', 'layer', 0.25, 0.9, 0.8, '#6C5CE7', (2, 0, 1)),
                ModuleGene('memory', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.8, 0.9, '#55EFC4', (3, 0, 0.5)),
                ModuleGene('output_gate', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'recurrent_memory',
            'inductive_bias': 'temporal_integration'
        },
        4: {  # Parallel Hybrid (Multi-pathway)
            'name': 'Dual-Stream Processing',
            'modules': [
                ModuleGene('input', 'conv', base_size, 'relu', 'batch', 0.1, 1.0, 0.3, '#FF6B6B', (0, 0, 0)),
                ModuleGene('dorsal_1', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.5, '#FD79A8', (1, 1, 0)),
                ModuleGene('dorsal_2', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.9, 0.6, '#FDCB6E', (2, 1, 0)),
                ModuleGene('ventral_1', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.5, '#48DBFB', (1, -1, 0)),
                ModuleGene('ventral_2', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.9, 0.6, '#A29BFE', (2, -1, 0)),
                ModuleGene('integrate', 'mlp', base_size*4, 'swish', 'layer', 0.25, 0.8, 0.7, '#00B894', (3, 0, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'dual_pathway',
            'inductive_bias': 'specialized_processing'
        },
        5: {  # Graph Neural Architecture
            'name': 'Graph Relational Network',
            'modules': [
                ModuleGene('embed', 'mlp', base_size, 'gelu', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('graph_1', 'graph', base_size*2, 'gelu', 'layer', 0.15, 1.0, 0.7, '#A29BFE', (1, 0.5, 0.5)),
                ModuleGene('graph_2', 'graph', base_size*3, 'gelu', 'layer', 0.2, 0.9, 0.7, '#74B9FF', (2, -0.5, 0.5)),
                ModuleGene('graph_3', 'graph', base_size*3, 'gelu', 'layer', 0.2, 0.9, 0.7, '#00CEC9', (2, 0.5, -0.5)),
                ModuleGene('aggregate', 'attention', base_size*4, 'gelu', 'layer', 0.25, 0.8, 0.6, '#55EFC4', (3, 0, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'fully_connected_graph',
            'inductive_bias': 'relational_reasoning'
        }
    }
    
    # Use modulo to wrap around the defined forms if form_id is out of bounds.
    # This allows for a functionally "infinite" number of forms by reusing the 
    # base templates, which will then diverge through evolution.
    lookup_id = ((form_id - 1) % len(forms)) + 1
    form = forms[lookup_id]
    modules = form['modules']
    
    # Create connections based on topology
    connections = []
    
    if form['topology'] == 'hierarchical':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.7, 1.0)), 'excitatory', 0.01, 'hebbian'
            ))
            
    elif form['topology'] == 'residual_attention':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.8, 1.0)), 'excitatory', 0.005, 'stdp'
            ))
        # Add residual connections
        for i in range(1, len(modules) - 2):
            if 'attn' in modules[i].id:
                connections.append(ConnectionGene(
                    modules[i].id, modules[i+2].id,
                    float(np.random.uniform(0.3, 0.5)), 'excitatory', 0.02, 'static'
                ))
                
    elif form['topology'] == 'recurrent_memory':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.7, 0.9)), 'excitatory', 0.01, 'hebbian'
            ))
        # Recurrent connections
        for module in modules:
            if 'lstm' in module.id:
                connections.append(ConnectionGene(
                    module.id, module.id,
                    float(np.random.uniform(0.4, 0.6)), 'modulatory', 0.001, 'stdp'
                ))
                
    elif form['topology'] == 'dual_pathway':
        # Input to both pathways
        connections.append(ConnectionGene('input', 'dorsal_1', 0.8, 'excitatory', 0.01, 'hebbian'))
        connections.append(ConnectionGene('input', 'ventral_1', 0.8, 'excitatory', 0.01, 'hebbian'))
        # Within pathways
        connections.append(ConnectionGene('dorsal_1', 'dorsal_2', 0.9, 'excitatory', 0.005, 'stdp'))
        connections.append(ConnectionGene('ventral_1', 'ventral_2', 0.9, 'excitatory', 0.005, 'stdp'))
        # Convergence
        connections.append(ConnectionGene('dorsal_2', 'integrate', 0.7, 'excitatory', 0.02, 'hebbian'))
        connections.append(ConnectionGene('ventral_2', 'integrate', 0.7, 'excitatory', 0.02, 'hebbian'))
        connections.append(ConnectionGene('integrate', 'output', 0.9, 'excitatory', 0.01, 'stdp'))
        
    elif form['topology'] == 'fully_connected_graph':
        for i, m1 in enumerate(modules):
            for j, m2 in enumerate(modules):
                if i != j and np.random.random() > 0.4:
                    weight = float(np.random.uniform(0.3, 0.8) if i < j else np.random.uniform(0.2, 0.5))
                    conn_type = 'excitatory' if i < j else 'modulatory'
                    connections.append(ConnectionGene(
                        m1.id, m2.id, weight, conn_type,
                        float(np.random.uniform(0.001, 0.02)), 'hebbian'
                    ))
    
    # Create developmental rules
    dev_rules = [
        DevelopmentalGene('proliferation', 'fitness_plateau', {'growth_rate': 1.1, 'max_size': max_size * 2, 'stagnation_threshold': 3}),
        DevelopmentalGene('pruning', 'maturity', {'threshold': 0.1, 'maturity_age': 5}),
        DevelopmentalGene('differentiation', 'environmental_signal', {'specialization_strength': 0.7})
    ]
    
    genotype = Genotype(
        modules=modules,
        connections=connections,
        developmental_rules=dev_rules,
        form_id=form_id
    )
    genotype.complexity = genotype.compute_complexity()
    
    return genotype

# ==================== ADVANCED EVOLUTION ====================

def mutate(genotype: Genotype, mutation_rate: float = 0.2, innovation_rate: float = 0.05) -> Genotype:
    """Biologically-inspired mutation with innovation"""
    mutated = genotype.copy()
    mutated.age = 0
    
    # 1. Point mutations (module parameters)
    for module in mutated.modules:
        if random.random() < mutation_rate:
            # Size mutation with drift
            change_factor = np.random.lognormal(0, 0.2)
            module.size = int(module.size * change_factor)
            module.size = int(np.clip(module.size, 16, 1024))
        
        if random.random() < mutation_rate * 0.5:
            # Plasticity mutation
            module.plasticity += np.random.normal(0, 0.1)
            module.plasticity = float(np.clip(module.plasticity, 0, 1))
        
        if random.random() < mutation_rate * 0.3:
            # Learning rate multiplier
            module.learning_rate_mult *= np.random.lognormal(0, 0.15)
            module.learning_rate_mult = float(np.clip(module.learning_rate_mult, 0.1, 2.0))
        
        if random.random() < mutation_rate * 0.2:
            # Activation function mutation
            module.activation = random.choice(['relu', 'gelu', 'silu', 'swish'])
    
    # 2. Connection weight mutations
    for connection in mutated.connections:
        if random.random() < mutation_rate:
            connection.weight += np.random.normal(0, 0.15)
            connection.weight = float(np.clip(connection.weight, 0.05, 1.0))
        
        if random.random() < mutation_rate * 0.3:
            # Plasticity rule mutation
            connection.plasticity_rule = random.choice(['hebbian', 'anti-hebbian', 'stdp', 'static'])
    
    # 3. Structural mutations (innovation)
    if random.random() < innovation_rate:
        # Add new connection
        if len(mutated.modules) > 2:
            source = random.choice(mutated.modules[:-1])
            target = random.choice([m for m in mutated.modules if m.id != source.id])
            
            # Check if connection already exists
            exists = any(c.source == source.id and c.target == target.id for c in mutated.connections)
            if not exists:
                mutated.connections.append(ConnectionGene(
                    source.id, target.id,
                    float(np.random.uniform(0.2, 0.5)), 'excitatory',
                    float(np.random.uniform(0.001, 0.02)), 'hebbian'
                ))
    
    if random.random() < innovation_rate * 0.5:
        # Add new module (rare)
        new_id = f"evolved_{len(mutated.modules)}"
        avg_size = int(np.mean([m.size for m in mutated.modules]))
        new_module = ModuleGene(
            new_id, random.choice(['mlp', 'attention', 'graph']),
            avg_size, random.choice(['gelu', 'swish']), 'layer',
            0.2, 1.0, 0.5, '#DDA15E',
            (len(mutated.modules), 0, 0)
        )
        mutated.modules.append(new_module)
        
        # Connect to network
        source = random.choice(mutated.modules[:-1])
        mutated.connections.append(ConnectionGene(
            source.id, new_id, 0.3, 'excitatory', 0.01, 'hebbian'
        ))
    
    # 4. Meta-parameter mutations
    for key in mutated.meta_parameters:
        if random.random() < mutation_rate * 0.4:
            mutated.meta_parameters[key] *= np.random.lognormal(0, 0.1)
    
    mutated.complexity = mutated.compute_complexity()
    return mutated

def crossover(parent1: Genotype, parent2: Genotype, crossover_rate: float = 0.7) -> Genotype:
    """Advanced recombination with homologous alignment"""
    if parent1.form_id != parent2.form_id:
        return parent1.copy()
    
    child = parent1.copy()
    child.parent_ids = [parent1.lineage_id, parent2.lineage_id]
    
    if random.random() > crossover_rate:
        return child
    
    # Epigenetic Crossover: average the decayed markers from both parents
    # child.epigenetic_markers already contains decayed markers from parent1 via .copy()
    for key, p2_val in parent2.epigenetic_markers.items():
        # Decay parent2's markers as well before averaging
        decayed_p2_val = p2_val * 0.5 
        # Get the current value (from parent1) and average with parent2's
        current_val = child.epigenetic_markers.get(key, 0.0)
        child.epigenetic_markers[key] = (current_val + decayed_p2_val) / 2.0

    # Module-level crossover
    for i in range(min(len(parent1.modules), len(parent2.modules))):
        if random.random() < 0.5:
            # Inherit from parent2
            child.modules[i].size = parent2.modules[i].size
            child.modules[i].plasticity = parent2.modules[i].plasticity
            child.modules[i].learning_rate_mult = parent2.modules[i].learning_rate_mult
    
    # Connection-level crossover
    p1_conns = {(c.source, c.target): c for c in parent1.connections}
    p2_conns = {(c.source, c.target): c for c in parent2.connections}
    
    new_connections = []
    all_keys = set(p1_conns.keys()) | set(p2_conns.keys())
    
    for key in all_keys:
        if key in p1_conns and key in p2_conns:
            # Both parents have this connection
            conn = p1_conns[key] if random.random() < 0.5 else p2_conns[key]
            new_connections.append(ConnectionGene(
                conn.source, conn.target, conn.weight, conn.connection_type,
                conn.delay, conn.plasticity_rule
            ))
        elif key in p1_conns:
            new_connections.append(ConnectionGene(
                p1_conns[key].source, p1_conns[key].target, p1_conns[key].weight,
                p1_conns[key].connection_type, p1_conns[key].delay, p1_conns[key].plasticity_rule
            ))
        else:
            new_connections.append(ConnectionGene(
                p2_conns[key].source, p2_conns[key].target, p2_conns[key].weight,
                p2_conns[key].connection_type, p2_conns[key].delay, p2_conns[key].plasticity_rule
            ))
    
    child.connections = new_connections
    
    # Meta-parameter crossover
    for key in child.meta_parameters:
        if key in parent2.meta_parameters and random.random() < 0.5:
            child.meta_parameters[key] = parent2.meta_parameters[key]
    
    child.complexity = child.compute_complexity()
    return child

def apply_endosymbiosis(recipient: Genotype, donors: List[Genotype]) -> Genotype:
    """
    A rare event where a recipient genotype acquires a module from a highly fit donor.
    This simulates horizontal gene transfer or endosymbiosis.
    """
    if not donors or not recipient.connections:
        return recipient

    # Select a random elite donor
    donor = random.choice(donors)
    
    # Select a non-trivial module from the donor to acquire
    candidate_modules = [m for m in donor.modules if m.module_type not in ['input', 'output']]
    if not candidate_modules:
        return recipient

    module_to_acquire = ModuleGene(**asdict(random.choice(candidate_modules)))
    
    # Ensure the new module has a unique ID
    new_id = f"endo_{module_to_acquire.id}_{random.randint(0, 999)}"
    if any(m.id == new_id for m in recipient.modules):
        return recipient # Avoid ID collision
    module_to_acquire.id = new_id

    # Insert the module by splitting an existing connection
    connection_to_split = random.choice(recipient.connections)
    recipient.connections.remove(connection_to_split)

    # Add the new module and wire it in
    recipient.modules.append(module_to_acquire)
    recipient.connections.append(ConnectionGene(connection_to_split.source, new_id, float(np.random.uniform(0.4, 0.8)), 'excitatory', 0.01, 'hebbian'))
    recipient.connections.append(ConnectionGene(new_id, connection_to_split.target, float(np.random.uniform(0.4, 0.8)), 'excitatory', 0.01, 'stdp'))

    recipient.complexity = recipient.compute_complexity()
    st.toast(f"Endosymbiosis Event: Acquired module '{module_to_acquire.module_type}'.", icon="🧬")
    return recipient

def apply_developmental_rules(genotype: Genotype, stagnation_counter: int) -> Genotype:
    """
    Executes the developmental program encoded in the genotype.
    This simulates processes like pruning and proliferation during an individual's life.
    """
    developed_genotype = genotype # Work on the same object
    
    for rule in developed_genotype.developmental_rules:
        if rule.rule_type == 'pruning' and rule.trigger_condition == 'maturity':
            # Prune weak connections if the individual is mature enough
            if developed_genotype.age > rule.parameters.get('maturity_age', 5):
                threshold = rule.parameters.get('threshold', 0.1)
                developed_genotype.connections = [
                    c for c in developed_genotype.connections if c.weight >= threshold
                ]
                
        elif rule.rule_type == 'proliferation' and rule.trigger_condition == 'fitness_plateau':
            # Grow a module if the population is stagnating
            if stagnation_counter > rule.parameters.get('stagnation_threshold', 3):
                if developed_genotype.modules:
                    target_module = random.choice(developed_genotype.modules)
                    growth_rate = rule.parameters.get('growth_rate', 1.1)
                    max_size = rule.parameters.get('max_size', 2048)
                    target_module.size = int(min(target_module.size * growth_rate, max_size))

    # Recalculate complexity after development
    developed_genotype.complexity = developed_genotype.compute_complexity()
    return developed_genotype

def evaluate_fitness(genotype: Genotype, task_type: str, generation: int, weights: Optional[Dict[str, float]] = None, enable_epigenetics: bool = False, enable_baldwin: bool = False, epistatic_linkage_k: int = 0, parasite_profile: Optional[Dict] = None) -> Tuple[float, Dict[str, float]]:
    """
    Multi-objective fitness evaluation with realistic task simulation
    
    Returns: (total_fitness, component_scores)
    """
    
    # Initialize component scores
    scores = {
        'task_accuracy': 0.0,
        'efficiency': 0.0,
        'robustness': 0.0,
        'generalization': 0.0
    }
    
    # Compute architectural properties
    total_params = sum(m.size for m in genotype.modules)
    avg_plasticity = np.mean([m.plasticity for m in genotype.modules])
    connection_density = len(genotype.connections) / (len(genotype.modules) ** 2 + 1)
    
    # 1a. Epigenetic Inheritance Bonus
    # Apply bonus from markers inherited from parents.
    epigenetic_bonus = 0.0
    if enable_epigenetics:
        aptitude_key = f"{task_type}_aptitude"
        if aptitude_key in genotype.epigenetic_markers:
            epigenetic_bonus = genotype.epigenetic_markers[aptitude_key]
    
    # 1b. Task-specific accuracy simulation
    if task_type == 'Abstract Reasoning (ARC-AGI-2)':
        # Reward compositional structures and high plasticity
        graph_attention_count = sum(1 for m in genotype.modules 
                                   if m.module_type in ['graph', 'attention'])
        compositional_score = graph_attention_count / len(genotype.modules)
        plasticity_bonus = avg_plasticity * 0.4
        
        # ARC requires efficiency and abstraction
        efficiency_penalty = np.exp(-total_params / 50000)  # Prefer compact architectures
        
        scores['task_accuracy'] = (
            compositional_score * 0.4 +
            plasticity_bonus * 0.3 +
            efficiency_penalty * 0.3 +
            np.random.normal(0, 0.05)  # Stochasticity
        )
        
        # Evolved architectures improve over generations
        scores['task_accuracy'] *= (1 + 0.01 * generation)
        
    elif task_type == 'Vision (ImageNet)':
        conv_count = sum(1 for m in genotype.modules if m.module_type == 'conv')
        hierarchical_bonus = 0.2 if genotype.form_id in [1, 4] else 0.0
        
        scores['task_accuracy'] = (
            (conv_count / len(genotype.modules)) * 0.5 +
            hierarchical_bonus +
            connection_density * 0.2 +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Language (MMLU-Pro)':
        attn_count = sum(1 for m in genotype.modules if m.module_type == 'attention')
        depth_bonus = len(genotype.modules) / 10
        
        scores['task_accuracy'] = (
            (attn_count / len(genotype.modules)) * 0.6 +
            min(depth_bonus, 0.3) +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Sequential Prediction':
        rec_count = sum(1 for m in genotype.modules if m.module_type == 'recurrent')
        memory_bonus = 0.3 if any('memory' in m.id for m in genotype.modules) else 0.0
        
        scores['task_accuracy'] = (
            (rec_count / len(genotype.modules)) * 0.5 +
            memory_bonus +
            avg_plasticity * 0.15 +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Multi-Task Learning':
        module_diversity = len(set(m.module_type for m in genotype.modules))
        hybrid_bonus = 0.4 if genotype.form_id in [4, 5] else 0.0
        
        scores['task_accuracy'] = (
            (module_diversity / 5) * 0.4 +
            hybrid_bonus +
            connection_density * 0.15 +
            np.random.normal(0, 0.05)
        )
    
    # Apply epigenetic bonus to the base score
    scores['task_accuracy'] += epigenetic_bonus

    # 2. Lifetime Learning Simulation (Baldwin Effect)
    # Plasticity allows an individual to "learn" and improve its performance during its lifetime.
    # This bonus is added to the base task accuracy, rewarding adaptable architectures.
    if enable_baldwin:
        lifetime_learning_bonus = avg_plasticity * 0.2  # Max 20% accuracy boost from learning
        scores['task_accuracy'] += lifetime_learning_bonus
    
    # Clamp task accuracy after adding bonus
    scores['task_accuracy'] = np.clip(scores['task_accuracy'], 0, 1)
    
    # 3. Efficiency score (inverse of computational cost)
    # Prefer architectures with good accuracy-to-parameter ratio
    param_efficiency = 1.0 / (1.0 + np.log(1 + total_params / 10000))
    connection_efficiency = 1.0 - min(connection_density, 0.8)
    
    scores['efficiency'] = (param_efficiency + connection_efficiency) / 2
    
    # 4. Robustness (architectural stability)
    # More diverse connections and moderate plasticity = more robust
    robustness_from_diversity = len(set(c.connection_type for c in genotype.connections)) / 3
    robustness_from_plasticity = 1.0 - abs(avg_plasticity - 0.5) * 2  # Prefer moderate
    
    scores['robustness'] = (robustness_from_diversity * 0.5 + robustness_from_plasticity * 0.5)
    
    # 5. Generalization potential
    # Architectural properties that predict generalization
    depth = len(genotype.modules)
    modularity_score = 1.0 - abs(connection_density - 0.3) * 2  # Sweet spot at 0.3
    
    scores['generalization'] = (
        min(depth / 10, 1.0) * 0.4 +
        modularity_score * 0.3 +
        avg_plasticity * 0.3
    )

    # 6. Epigenetic Marking (Lamarckian-like learning)
    # The individual "learns" from its performance, creating a marker for its offspring.
    # This maps current performance to a small, heritable aptitude value.
    if enable_epigenetics:
        aptitude_key = f"{task_type}_aptitude"
        performance_marker = (scores['task_accuracy'] - 0.5) * 0.05 # Small learning step
        current_aptitude = genotype.epigenetic_markers.get(aptitude_key, 0.0)
        genotype.epigenetic_markers[aptitude_key] = np.clip(current_aptitude + performance_marker, -0.15, 0.15)
    
    # 7. Epistatic Contribution (NK Landscape Simulation)
    # Models how gene interactions create a rugged fitness landscape.
    epistatic_contribution = 0.0
    if epistatic_linkage_k > 0 and len(genotype.modules) > epistatic_linkage_k:
        num_modules = len(genotype.modules)
        for i, module in enumerate(genotype.modules):
            # Select K interacting genes (modules)
            indices = list(range(num_modules))
            indices.remove(i)
            interacting_indices = random.sample(indices, k=epistatic_linkage_k)
            
            # Create a unique "genetic context" signature
            context_signature = tuple([module.module_type] + [genotype.modules[j].module_type for j in interacting_indices])
            
            # Use a hash to create a deterministic, pseudo-random contribution for this context
            hash_val = hash(context_signature)
            epistatic_contribution += (hash_val % 2000 - 1000) / 10000.0 # Small contribution in [-0.1, 0.1]

    # Multi-objective fitness with task-dependent weights
    if weights is None:
        if 'ARC' in task_type:
            weights = {'task_accuracy': 0.5, 'efficiency': 0.3, 'robustness': 0.1, 'generalization': 0.1}
        else:
            weights = {'task_accuracy': 0.6, 'efficiency': 0.2, 'robustness': 0.1, 'generalization': 0.1}
        
    total_fitness = sum(scores[k] * weights[k] for k in weights)
    
    # Apply epistatic effect to final fitness
    total_fitness += epistatic_contribution
    
    # 8. Red Queen Coevolution (Parasite Attack)
    # If a genotype has a trait targeted by the co-evolving parasite, its fitness is penalized.
    if parasite_profile:
        vulnerability_score = 0.0
        target_activation = parasite_profile.get('target_activation')
        if target_activation:
            for module in genotype.modules:
                if module.activation == target_activation:
                    vulnerability_score += 0.1 # Each matching module increases vulnerability
        
        total_fitness *= (1.0 - min(vulnerability_score, 0.5)) # Max 50% fitness reduction

    # Apply a small fitness floor. This prevents complete zeros for viable but
    # poorly performing individuals, which can help maintain diversity and
    # prevent numerical issues in some selection schemes.
    total_fitness = max(total_fitness, 1e-6)

    # Store component scores
    genotype.accuracy = scores['task_accuracy']
    genotype.efficiency = scores['efficiency']
    genotype.robustness = scores['robustness']
    
    return total_fitness, scores

def synthesize_master_architecture(top_individuals: List[Genotype]) -> Optional[Genotype]:
    """
    Synthesizes a "master" architecture by creating a consensus from the top n individuals.
    It starts with the best individual and refines its parameters by averaging them
    with other elite individuals. It may also add highly-voted structural elements.
    """
    if not top_individuals:
        return None

    n = len(top_individuals)
    best_ind = top_individuals[0]
    
    # Start with a copy of the best individual as the template
    master_arch = best_ind.copy()
    master_arch.lineage_id = "SYNTHESIZED_MASTER"
    master_arch.fitness = float(np.mean([ind.fitness for ind in top_individuals]))
    master_arch.accuracy = float(np.mean([ind.accuracy for ind in top_individuals]))
    master_arch.efficiency = float(np.mean([ind.efficiency for ind in top_individuals]))
    master_arch.robustness = float(np.mean([ind.robustness for ind in top_individuals]))

    # --- 1. Parameter Averaging ---
    
    # Create lookups for faster access
    all_modules_by_id = {ind.lineage_id: {m.id: m for m in ind.modules} for ind in top_individuals}
    all_conns_by_key = {ind.lineage_id: {(c.source, c.target): c for c in ind.connections} for ind in top_individuals}

    # Average module parameters
    for master_module in master_arch.modules:
        module_id = master_module.id
        peers = [all_modules_by_id[ind.lineage_id].get(module_id) for ind in top_individuals]
        peers = [p for p in peers if p is not None]
        
        if len(peers) > 1:
            master_module.size = int(np.mean([p.size for p in peers]))
            master_module.dropout_rate = float(np.mean([p.dropout_rate for p in peers]))
            master_module.learning_rate_mult = float(np.mean([p.learning_rate_mult for p in peers]))
            master_module.plasticity = float(np.mean([p.plasticity for p in peers]))

    # Average connection parameters
    for master_conn in master_arch.connections:
        conn_key = (master_conn.source, master_conn.target)
        peers = [all_conns_by_key[ind.lineage_id].get(conn_key) for ind in top_individuals]
        peers = [p for p in peers if p is not None]
        
        if len(peers) > 1:
            master_conn.weight = float(np.mean([p.weight for p in peers]))
            plasticity_rules = [p.plasticity_rule for p in peers]
            master_conn.plasticity_rule = Counter(plasticity_rules).most_common(1)[0][0]

    # --- 2. Structural Voting (Add missing consensus connections) ---
    connection_counts = Counter()
    all_connections_map = {} 
    for ind in top_individuals:
        for conn in ind.connections:
            key = (conn.source, conn.target)
            connection_counts[key] += 1
            if key not in all_connections_map:
                all_connections_map[key] = conn

    master_conn_keys = {(c.source, c.target) for c in master_arch.connections}
    consensus_threshold = n / 2.0
    
    for conn_key, count in connection_counts.items():
        if count > consensus_threshold and conn_key not in master_conn_keys:
            source_id, target_id = conn_key
            master_module_ids = {m.id for m in master_arch.modules}
            if source_id in master_module_ids and target_id in master_module_ids:
                template_conn = all_connections_map[conn_key]
                peers = [all_conns_by_key[ind.lineage_id].get(conn_key) for ind in top_individuals]
                peers = [p for p in peers if p is not None]
                
                if not peers: continue

                avg_weight = float(np.mean([p.weight for p in peers]))
                plasticity_rule = Counter([p.plasticity_rule for p in peers]).most_common(1)[0][0]
                
                new_conn = ConnectionGene(
                    source=template_conn.source,
                    target=template_conn.target,
                    weight=avg_weight,
                    connection_type=template_conn.connection_type,
                    delay=template_conn.delay,
                    plasticity_rule=plasticity_rule
                )
                master_arch.connections.append(new_conn)
                
    master_arch.complexity = master_arch.compute_complexity()
    return master_arch

def analyze_lesion_sensitivity(
    master_architecture: Genotype, 
    base_fitness: float, 
    task_type: str, 
    fitness_weights: Dict, 
    eval_params: Dict
) -> Dict[str, float]:
    """
    Performs a lesion study on the master architecture to find critical components.
    Returns a dictionary of component ID to fitness drop (criticality).
    """
    criticality_scores = {}

    # 1. Lesion Modules
    for module in master_architecture.modules:
        # Don't lesion input/output
        if 'input' in module.id or 'output' in module.id:
            continue

        lesioned_arch = master_architecture.copy()
        
        # Remove the module and any connections to/from it
        lesioned_arch.modules = [m for m in lesioned_arch.modules if m.id != module.id]
        lesioned_arch.connections = [
            c for c in lesioned_arch.connections if c.source != module.id and c.target != module.id
        ]
        
        if not lesioned_arch.connections or not lesioned_arch.modules: continue # Skip if network is disconnected

        lesioned_fitness, _ = evaluate_fitness(
            lesioned_arch, 
            task_type, 
            lesioned_arch.generation, 
            fitness_weights, 
            **eval_params
        )
        
        fitness_drop = base_fitness - lesioned_fitness
        criticality_scores[f"Module: {module.id}"] = fitness_drop

    # 2. Lesion a few critical connections (highest weight)
    sorted_connections = sorted(master_architecture.connections, key=lambda c: c.weight, reverse=True)
    for conn in sorted_connections[:3]: # Lesion top 3 connections
        lesioned_arch = master_architecture.copy()
        
        lesioned_arch.connections = [c for c in lesioned_arch.connections if not (c.source == conn.source and c.target == conn.target)]
        
        lesioned_fitness, _ = evaluate_fitness(
            lesioned_arch, 
            task_type, 
            lesioned_arch.generation, 
            fitness_weights, 
            **eval_params
        )
        
        fitness_drop = base_fitness - lesioned_fitness
        criticality_scores[f"Conn: {conn.source}→{conn.target}"] = fitness_drop
        
    return criticality_scores

def analyze_information_flow(master_architecture: Genotype) -> Dict[str, float]:
    """
    Analyzes the flow of information through the network using graph centrality.
    Returns a dictionary of module ID to its betweenness centrality score.
    """
    G = nx.DiGraph()
    for module in master_architecture.modules: G.add_node(module.id)
    for conn in master_architecture.connections:
        if conn.weight > 1e-6: G.add_edge(conn.source, conn.target, weight=1.0/conn.weight)
    return nx.betweenness_centrality(G, weight='weight', normalized=True)

def analyze_evolvability_robustness(
    master_architecture: Genotype,
    task_type: str,
    fitness_weights: Dict,
    eval_params: Dict,
    num_mutants: int = 50
) -> Dict:
    """
    Analyzes the trade-off between robustness (resistance to mutation) and
    evolvability (potential for beneficial mutation).
    """
    base_fitness = master_architecture.fitness
    fitness_changes = []

    for _ in range(num_mutants):
        mutant = mutate(master_architecture, mutation_rate=0.1, innovation_rate=0.02)
        mutant_fitness, _ = evaluate_fitness(
            mutant, task_type, mutant.generation, fitness_weights, **eval_params
        )
        fitness_changes.append(mutant_fitness - base_fitness)
    
    fitness_changes = np.array(fitness_changes)
    
    robustness = -np.mean(fitness_changes[fitness_changes < 0]) if (fitness_changes < 0).any() else 0
    evolvability = np.max(fitness_changes) if (fitness_changes > 0).any() else 0
    
    return {
        "robustness": robustness,
        "evolvability": evolvability,
        "distribution": fitness_changes
    }

def analyze_developmental_trajectory(master_architecture: Genotype, steps: int = 20) -> pd.DataFrame:
    """
    Simulates the developmental program of the master architecture over a lifetime
    to see how its complexity changes.
    """
    trajectory = []
    arch = master_architecture.copy()
    
    for step in range(steps):
        arch.age = step + 1
        # Simulate stagnation by toggling it
        stagnation_counter = 5 if (step // 5) % 2 == 1 else 0
        
        # Store pre-development state
        pre_params = sum(m.size for m in arch.modules)
        pre_conns = len(arch.connections)
        
        arch = apply_developmental_rules(arch, stagnation_counter)
        
        # Store post-development state
        post_params = sum(m.size for m in arch.modules)
        post_conns = len(arch.connections)
        
        trajectory.append({
            "step": step,
            "total_params": post_params,
            "num_connections": post_conns,
            "pruned": pre_conns - post_conns > 0,
            "proliferated": post_params - pre_params > 0
        })
        
    return pd.DataFrame(trajectory)

def analyze_genetic_load(criticality_scores: Dict) -> Dict:
    """
    Identifies neutral or near-neutral components ("junk DNA") and calculates
    the genetic load they impose on the architecture.
    """
    neutral_threshold = 0.001 # Fitness drop less than this is considered neutral
    
    neutral_components = [
        comp for comp, drop in criticality_scores.items() if abs(drop) < neutral_threshold
    ]
    
    deleterious_components = [
        drop for comp, drop in criticality_scores.items() if drop < -neutral_threshold
    ]
    
    # Genetic load is the fitness reduction from slightly deleterious mutations
    genetic_load = -sum(deleterious_components)
    
    return {
        "neutral_component_count": len(neutral_components),
        "genetic_load": genetic_load
    }

def analyze_phylogenetic_signal(history_df: pd.DataFrame, final_population: List[Genotype]) -> Optional[Dict]:
    """
    Measures the correlation between phylogenetic distance and phenotypic distance.
    A high correlation means closely related individuals have similar traits.
    """
    # This is a placeholder for a more complex analysis.
    # A full implementation would require building a phylogenetic tree and calculating patristic distances.
    # We can simulate a result for demonstration.
    if len(final_population) < 10: return None
    
    # Simulate a plausible correlation
    base_corr = np.clip(final_population[0].fitness, 0.2, 0.8)
    phylo_dist = np.random.rand(45) * 10
    pheno_dist = phylo_dist * base_corr * np.random.uniform(0.5, 1.5, 45) + np.random.rand(45) * (1-base_corr)
    
    corr, _ = pearsonr(phylo_dist, pheno_dist)

    return {
        "correlation": corr,
        "phylo_distances": phylo_dist,
        "pheno_distances": pheno_dist
    }

def generate_pytorch_code(architecture: Genotype) -> str:
    """Generates a PyTorch nn.Module class from a genotype."""
    
    module_defs = []
    for module in architecture.modules:
        if module.module_type == 'mlp':
            # Assuming input size is same as output size for simplicity
            module_defs.append(f"            '{module.id}': nn.Sequential(nn.Linear({module.size}, {module.size}), nn.GELU()),")
        elif module.module_type == 'attention':
            module_defs.append(f"            '{module.id}': nn.MultiheadAttention(embed_dim={module.size}, num_heads=8, batch_first=True),")
        elif module.module_type == 'conv':
            module_defs.append(f"            '{module.id}': nn.Conv2d(in_channels=3, out_channels={module.size}, kernel_size=3, padding=1), # Assuming 3 input channels")
        elif module.module_type == 'recurrent':
            module_defs.append(f"            '{module.id}': nn.LSTM(input_size={module.size}, hidden_size={module.size}, batch_first=True),")
        else: # graph, etc.
            module_defs.append(f"            '{module.id}': nn.Identity(), # Placeholder for '{module.module_type}'")

    module_defs_str = "\n".join(module_defs)

    # Create a simplified topological sort for the forward pass
    G = nx.DiGraph()
    for conn in architecture.connections: G.add_edge(conn.source, conn.target)
    try:
        exec_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible: # Handle cycles
        # Fallback for cyclic graphs: just use module order and hope for the best
        exec_order = [m.id for m in architecture.modules]

    forward_pass = ["        outputs = {'input': x} # Assuming 'input' is the first module's ID"]
    for module_id in exec_order:
        # Find inputs for the current module
        inputs = [c.source for c in architecture.connections if c.target == module_id]
        if not inputs:
            if module_id != 'input': forward_pass.append(f"        # Module '{module_id}' has no inputs, skipping.")
            continue
        
        # Simple aggregation: sum inputs
        input_str = " + ".join([f"outputs['{i}']" for i in inputs])
        
        # Handle special cases for nn.Module outputs (e.g., LSTM)
        if any(m.module_type == 'recurrent' and m.id == module_id for m in architecture.modules):
            forward_pass.append(f"        out, _ = self.evolved_modules['{module_id}']({input_str})")
            forward_pass.append(f"        outputs['{module_id}'] = out")
        elif any(m.module_type == 'attention' and m.id == module_id for m in architecture.modules):
             forward_pass.append(f"        attn_out, _ = self.evolved_modules['{module_id}']({input_str}, {input_str}, {input_str})")
             forward_pass.append(f"        outputs['{module_id}'] = attn_out")
        else:
            forward_pass.append(f"        outputs['{module_id}'] = self.evolved_modules['{module_id}']({input_str})")

    forward_pass.append("        return outputs['output'] # Assuming 'output' is the final module's ID")
    forward_pass_str = "\n".join(forward_pass)

    code = f"""
import torch
import torch.nn as nn

class EvolvedArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.evolved_modules = nn.ModuleDict({{
{module_defs_str}
        }})

    def forward(self, x):
{forward_pass_str}
"""
    return code.strip()

def generate_tensorflow_code(architecture: Genotype) -> str:
    """Generates a TensorFlow/Keras tf.keras.Model class from a genotype."""
    
    module_defs = []
    for module in architecture.modules:
        if module.module_type == 'mlp':
            module_defs.append(f"        self.{module.id} = tf.keras.Sequential([tf.keras.layers.Dense({module.size}, activation='gelu')])")
        elif module.module_type == 'attention':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim={module.size})")
        elif module.module_type == 'conv':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.Conv2D(filters={module.size}, kernel_size=3, padding='same', activation='relu')")
        elif module.module_type == 'recurrent':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.LSTM(units={module.size}, return_sequences=True)")
        else: # graph, etc.
            module_defs.append(f"        self.{module.id} = tf.keras.layers.Layer(name='{module.id}') # Placeholder for '{module.module_type}'")

    module_defs_str = "\n".join(module_defs)

    # Topological sort for the call pass
    G = nx.DiGraph()
    for conn in architecture.connections: G.add_edge(conn.source, conn.target)
    try:
        exec_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible: # Handle cycles
        exec_order = [m.id for m in architecture.modules]

    call_pass = ["        outputs = {{'input': inputs}} # Assuming 'input' is the first module's ID"]
    for module_id in exec_order:
        inputs = [c.source for c in architecture.connections if c.target == module_id]
        if not inputs:
            if module_id != 'input': call_pass.append(f"        # Module '{module_id}' has no inputs, skipping.")
            continue
        
        # Simple aggregation: sum inputs if more than one
        if len(inputs) > 1:
            input_str = " + ".join([f"outputs['{i}']" for i in inputs])
            call_pass.append(f"        aggregated_input = {input_str}")
            current_input = "aggregated_input"
        else:
            current_input = f"outputs['{inputs[0]}']"

        if any(m.module_type == 'attention' and m.id == module_id for m in architecture.modules):
             call_pass.append(f"        outputs['{module_id}'] = self.{module_id}(query={current_input}, value={current_input}, key={current_input})")
        else:
            call_pass.append(f"        outputs['{module_id}'] = self.{module_id}({current_input})")

    call_pass.append("        return outputs['output'] # Assuming 'output' is the final module's ID")
    call_pass_str = "\n".join(call_pass)

    code = f"""
import tensorflow as tf

class EvolvedArchitecture(tf.keras.Model):
    def __init__(self):
        super().__init__()
{module_defs_str}

    def call(self, inputs):
{call_pass_str}
"""
    return code.strip()

def identify_pareto_frontier(individuals: List[Genotype]) -> List[Genotype]:
    """
    Identifies the Pareto frontier from a list of individuals based on multiple objectives.
    Objectives are: accuracy, efficiency, robustness (all to be maximized).
    """
    if not individuals:
        return []

    pareto_frontier_indices = []
    
    for i, p in enumerate(individuals):
        is_dominated = False
        for j, q in enumerate(individuals):
            if i == j:
                continue
            
            # Check if q dominates p
            # q must be better or equal on all objectives, and strictly better on at least one.
            if (q.accuracy >= p.accuracy and q.efficiency >= p.efficiency and q.robustness >= p.robustness) and \
               (q.accuracy > p.accuracy or q.efficiency > p.efficiency or q.robustness > p.robustness):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier_indices.append(i)
            
    return [individuals[i] for i in pareto_frontier_indices]

# ==================== VISUALIZATION ====================

def visualize_fitness_landscape(history_df: pd.DataFrame):
    """Renders a highly comprehensive 3D fitness landscape with multiple evolutionary trajectories and analyses."""
    st.markdown("### The Fitness Landscape: A Multi-Dimensional View of the Evolutionary Search")
    st.markdown("""
    This visualization provides a deep, multi-faceted view into the evolutionary process, modeling the fitness landscape and the population's journey across it. It is composed of several key elements:

    - **The Fitness Surface (Z-axis):** This semi-transparent surface represents the *estimated* fitness landscape, where height (Z-axis) corresponds to the mean fitness of genotypes found within a given region of the search space. The axes of this space are two fundamental phenotypic traits: **Total Parameters** (a measure of size, on a log scale) and **Architectural Complexity** (a structural measure). A rugged, mountainous surface indicates a complex problem with many local optima, while a smooth hill suggests a simpler optimization task.

    - **Population Mean Trajectory (Thick Line):** This line tracks the average genotype of the entire population over generations. Its path reveals the central tendency of the evolutionary search. A steady climb indicates consistent progress, while wandering or stagnation points to challenges in finding better solutions.

    - **Apex Trajectory (Bright Line):** This line follows the path of the *fittest individual* from each generation. It represents the "leading edge" of evolution, showing the breakthroughs and discoveries made by the most successful lineages. The divergence between the Mean and Apex trajectories highlights the difference between the population average and the high-performing outliers that drive progress.

    - **Final Population Scatter (Points):** The individual points represent every member of the final generation, positioned according to their traits and colored by their fitness. This scatter plot reveals the final state of the population:
        - A tight cluster indicates **convergent evolution**, where the population has honed in on a specific optimal design.
        - A widely dispersed cloud suggests **divergent evolution**, with multiple, distinct solutions coexisting on a Pareto frontier.

    - **Trajectory Projections:** The shadows on the "walls" of the plot show the 2D projection of the trajectories, helping to isolate the movement along each pair of axes.
    """)

    # Use a subset for performance if history is large, ensuring we have data
    sample_size = min(len(history_df), 20000)
    if sample_size < 20: # Increased threshold for better surface
        st.warning("Not enough data to render a detailed fitness landscape.")
        return
    df_sample = history_df.sample(n=sample_size, random_state=42)
    
    x_param = 'total_params'
    y_param = 'complexity'
    z_param = 'fitness'

    # --- 1. Create the Fitness Surface ---
    # Create grid for the surface
    x_min_val = df_sample[x_param].min()
    x_max_val = df_sample[x_param].max()
    if x_min_val <= 0: x_min_val = 1
    if x_max_val <= x_min_val: x_max_val = x_min_val * 10

    x_bins = np.logspace(np.log10(x_min_val), np.log10(x_max_val), 30)
    y_bins = np.linspace(df_sample[y_param].min(), df_sample[y_param].max(), 30)

    # Bin data and calculate mean fitness for each grid cell
    df_sample['x_bin'] = pd.cut(df_sample[x_param], bins=x_bins, labels=False, include_lowest=True)
    df_sample['y_bin'] = pd.cut(df_sample[y_param], bins=y_bins, labels=False, include_lowest=True)
    grid = df_sample.groupby(['x_bin', 'y_bin'])[z_param].mean().unstack(level='x_bin')
    
    # Get grid coordinates (bin centers)
    x_coords = (x_bins[:-1] + x_bins[1:]) / 2
    y_coords = (y_bins[:-1] + y_bins[1:]) / 2
    z_surface = grid.values

    surface_trace = go.Surface(
        x=np.log10(x_coords), 
        y=y_coords, 
        z=z_surface,
        colorscale='cividis',
        opacity=0.6,
        colorbar=dict(title='Mean Fitness', x=1.0, len=0.7),
        name='Estimated Fitness Landscape',
        hoverinfo='x+y+z',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
    )

    # --- 2. Calculate Evolutionary Trajectories ---
    # Mean trajectory
    mean_trajectory = history_df.groupby('generation').agg({
        x_param: 'mean',
        y_param: 'mean',
        z_param: 'mean'
    }).reset_index()

    # Best-of-generation (Apex) trajectory
    apex_trajectory = history_df.loc[history_df.groupby('generation')['fitness'].idxmax()]

    # --- 3. Create Trajectory Traces ---
    mean_trajectory_trace = go.Scatter3d(
        x=np.log10(mean_trajectory[x_param]),
        y=mean_trajectory[y_param],
        z=mean_trajectory[z_param],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.8)', width=10),
        name='Population Mean Trajectory',
        hovertext=[f"Gen: {g}<br>Mean Fitness: {f:.3f}" for g, f in zip(mean_trajectory['generation'], mean_trajectory[z_param])],
        hoverinfo='text+name',
        projection=dict(x=dict(show=True), y=dict(show=True), z=dict(show=True))
    )

    apex_trajectory_trace = go.Scatter3d(
        x=np.log10(apex_trajectory[x_param]),
        y=apex_trajectory[y_param],
        z=apex_trajectory[z_param],
        mode='lines+markers',
        line=dict(color='cyan', width=5, dash='dot'),
        marker=dict(size=4, color='cyan'),
        name='Apex (Best) Trajectory',
        hovertext=[f"Gen: {g}<br>Best Fitness: {f:.3f}" for g, f in zip(apex_trajectory['generation'], apex_trajectory[z_param])],
        hoverinfo='text+name',
        projection=dict(x=dict(show=True), y=dict(show=True), z=dict(show=True))
    )

    # --- 4. Create Final Population Scatter ---
    final_gen_df = history_df[history_df['generation'] == history_df['generation'].max()]
    
    final_pop_trace = go.Scatter3d(
        x=np.log10(final_gen_df[x_param].clip(lower=1)),
        y=final_gen_df[y_param],
        z=final_gen_df[z_param],
        mode='markers',
        marker=dict(
            size=final_gen_df['accuracy'] * 10 + 2,
            color=final_gen_df['fitness'],
            colorscale='Viridis',
            colorbar=dict(title='Final Fitness', x=1.15, len=0.7),
            showscale=True,
            sizemin=4,
            sizemode='diameter'
        ),
        name='Final Population',
        hovertext=[
            f"Fitness: {f:.3f}<br>Accuracy: {a:.3f}<br>Params: {p:,.0f}<br>Form: {form}"
            for f, a, p, form in zip(final_gen_df['fitness'], final_gen_df['accuracy'], final_gen_df['total_params'], final_gen_df['form'])
        ],
        hoverinfo='text+name'
    )

    # --- 5. Assemble Figure and Update Layout ---
    fig = go.Figure(data=[surface_trace, mean_trajectory_trace, apex_trajectory_trace, final_pop_trace])
    
    fig.update_layout(
        title='<b>3D Fitness Landscape with Multi-Trajectory Analysis</b>',
        scene=dict(
            xaxis_title='Log(Total Parameters)',
            yaxis_title='Architectural Complexity',
            zaxis_title='Fitness',
            camera=dict(eye=dict(x=-1.9, y=-1.9, z=1.6)),
            aspectmode='cube' # Makes the plot a cube for better perspective
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    st.plotly_chart(fig, width='stretch', key="fitness_landscape_3d")

def visualize_phase_space_portraits(history_df: pd.DataFrame, metrics_df: pd.DataFrame):
    """
    Plots highly detailed phase-space portraits of key evolutionary dynamics,
    including velocity vectors to show system acceleration.
    """
    st.markdown("### Phase-Space Portraits: The Physics of Evolution")
    st.markdown("""
    This visualization, deeply rooted in **dynamical systems theory**, models the evolution as a particle moving through a multi-dimensional "phase space." Each plot represents a 2D slice of this space, showing a key system property versus its own rate of change. This transforms the generational data into a continuous trajectory, revealing the underlying "physics" of the evolutionary process.

    - **Points & Trajectory:** Each point is a generation, and the line connecting them shows the system's path through time.
    - **The `y=0` Line (Nullcline):** This is the line of equilibrium. If the trajectory crosses this line, the system's property stops changing at that instant.
    - **Acceleration Vectors (Arrows):** The small arrows attached to each point represent the system's "acceleration" vector `(d(X)/dt, d²(X)/dt²)`. They show the direction and magnitude of the "force" acting on the system at that moment, indicating where the trajectory is being pulled next.
    
    **How to Interpret the Dynamics:**
    - **Spiraling Inward:** If the trajectory spirals towards a point on the `y=0` line, it indicates a **stable equilibrium** or **attractor**. The system is converging to a stable state (e.g., peak fitness, optimal diversity).
    - **Spiraling Outward:** A trajectory spiraling away from a point indicates an **unstable equilibrium** or **repeller**. The system is actively moving away from that state.
    - **Closed Loops:** A repeating loop is a **limit cycle**, representing a stable, periodic behavior in the system (e.g., predator-prey dynamics between different strategies).
    - **Fast Transients:** Large vectors indicate periods of rapid change and instability, often following an environmental shift or a major innovation.
    """)

    # --- Data Preparation ---
    if len(metrics_df) < 3:
        st.info("Not enough generational data to compute phase-space dynamics (requires at least 3 generations).")
        return

    # Calculate additional metrics from history_df
    arch_stats = history_df.groupby('generation')[['complexity']].mean().reset_index()
    
    selection_diff_data = []
    for gen in sorted(history_df['generation'].unique()):
        gen_data = history_df[history_df['generation'] == gen]
        if len(gen_data) > 5:
            fitness_array = gen_data['fitness'].values
            num_survivors = max(2, int(len(gen_data) * 0.5)) # Assuming 50% pressure for this calc
            selected_idx = np.argpartition(fitness_array, -num_survivors)[-num_survivors:]
            diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
            selection_diff_data.append({'generation': gen, 'selection_diff': diff})
    
    sel_df = pd.DataFrame(selection_diff_data)

    # Merge all data
    df = metrics_df.merge(arch_stats, on='generation', how='left')
    if not sel_df.empty:
        df = df.merge(sel_df, on='generation', how='left')
    
    df = df.fillna(method='ffill').fillna(method='bfill') # Fill any gaps

    metrics_to_plot = {
        'mean_fitness': 'Mean Fitness (F)',
        'diversity': 'Genetic Diversity (H)',
        'complexity': 'Mean Complexity (C)',
        'selection_diff': 'Selection Pressure (S)'
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'<b>{v}</b> vs d/dt' for v in metrics_to_plot.values()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    plot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (metric, title), (r, c) in zip(metrics_to_plot.items(), plot_positions):
        if metric not in df.columns or df[metric].isnull().all():
            fig.add_annotation(text=f"Data for '{title}' not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=r, col=c)
            continue

        # Calculate derivatives
        x_data = df[metric]
        y_data = df[metric].diff()
        y_data_prime = y_data.diff() # Second derivative of the metric

        # Main trajectory trace
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines+markers',
            marker=dict(color=df['generation'], colorscale='plasma', showscale=(r==1 and c==1), colorbar=dict(title='Gen')),
            line=dict(color='rgba(128,128,128,0.3)'),
            hovertext=[f"Gen: {g}<br>{title.split(' ')[1]}: {x:.3f}<br>d/dt: {y:.3f}" for g, x, y in zip(df['generation'], x_data, y_data)],
            hoverinfo='text',
            name=title
        ), row=r, col=c)

        # Vector field (acceleration vectors)
        x_range = x_data.max() - x_data.min()
        y_range = y_data.max() - y_data.min()
        if len(df) > 2 and x_range > 1e-9 and y_range > 1e-9:
            arrow_len_fraction = 0.05
            arrow_len_x = x_range * arrow_len_fraction
            arrow_traces_x, arrow_traces_y = [], []

            for i in range(2, len(df)):
                x_pos, y_pos = x_data.iloc[i], y_data.iloc[i]
                u, v = y_data.iloc[i], y_data_prime.iloc[i]
                if pd.isna(u) or pd.isna(v): continue

                angle = np.arctan2(v / y_range, u / x_range)
                arrow_dx = arrow_len_x * np.cos(angle)
                arrow_dy = arrow_len_x * np.sin(angle) * (y_range / x_range)
                arrow_traces_x.extend([x_pos, x_pos + arrow_dx, None])
                arrow_traces_y.extend([y_pos, y_pos + arrow_dy, None])

            fig.add_trace(go.Scatter(x=arrow_traces_x, y=arrow_traces_y, mode='lines', line=dict(color='rgba(255,0,0,0.7)', width=1.5), hoverinfo='none', showlegend=False), row=r, col=c)

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=r, col=c)
        fig.update_xaxes(title_text=title, row=r, col=c)
        fig.update_yaxes(title_text=f"d({title.split('(')[1][0]})/dt", row=r, col=c)

    fig.update_layout(height=800, title_text="<b>Evolutionary Dynamics in Phase Space with Acceleration Vectors</b>", title_x=0.5, showlegend=False)
    st.plotly_chart(fig, width='stretch', key="phase_space_portraits_complex")

def visualize_genotype_3d(genotype: Genotype) -> go.Figure:
    """Advanced 3D network visualization"""
    
    # Create 3D node positions
    positions = {m.id: m.position for m in genotype.modules}
    
    # Create edges with color-coded types
    edge_traces = []
    edge_colors = {
        'excitatory': 'rgba(0, 255, 0, 0.6)',
        'inhibitory': 'rgba(255, 0, 0, 0.6)',
        'modulatory': 'rgba(0, 100, 255, 0.6)'
    }
    
    for conn in genotype.connections:
        if conn.source in positions and conn.target in positions:
            x0, y0, z0 = positions[conn.source]
            x1, y1, z1 = positions[conn.target]
            
            edge_traces.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(
                    width=conn.weight * 8,
                    color=edge_colors.get(conn.connection_type, 'rgba(125, 125, 125, 0.5)')
                ),
                hovertext=f'{conn.source}→{conn.target}<br>Weight: {conn.weight:.3f}<br>Type: {conn.connection_type}<br>Plasticity: {conn.plasticity_rule}',
                showlegend=False
            ))
    
    # Create nodes
    node_x = [m.position[0] for m in genotype.modules]
    node_y = [m.position[1] for m in genotype.modules]
    node_z = [m.position[2] for m in genotype.modules]
    node_colors = [m.color for m in genotype.modules]
    node_sizes = [m.size / 5 for m in genotype.modules]
    node_text = [
        f"<b>{m.id}</b><br>"
        f"Type: {m.module_type}<br>"
        f"Size: {m.size}<br>"
        f"Activation: {m.activation}<br>"
        f"Plasticity: {m.plasticity:.3f}<br>"
        f"LR Mult: {m.learning_rate_mult:.3f}"
        for m in genotype.modules
    ]
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=[m.id for m in genotype.modules],
        hovertext=node_text,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=f"<b>Form {genotype.form_id}</b> | Gen {genotype.generation} | Fitness: {genotype.fitness:.4f}<br>"
              f"<sub>Accuracy: {genotype.accuracy:.3f} | Efficiency: {genotype.efficiency:.3f} | "
              f"Complexity: {genotype.complexity:.3f} | Params: {sum(m.size for m in genotype.modules):,}</sub>",
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            yaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            zaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=500,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig

def visualize_genotype_2d(genotype: Genotype) -> go.Figure:
    """Creates a clear 2D visualization of a genotype for analysis."""
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for module in genotype.modules:
        G.add_node(
            module.id,
            size=module.size,
            color=module.color,
            module_type=module.module_type,
            hover_text=(
                f"<b>{module.id}</b><br>"
                f"Type: {module.module_type}<br>"
                f"Size: {module.size}<br>"
                f"Activation: {module.activation}<br>"
                f"Plasticity: {module.plasticity:.3f}"
            )
        )
        
    # Add edges with attributes
    for conn in genotype.connections:
        if conn.source in G.nodes and conn.target in G.nodes:
            G.add_edge(conn.source, conn.target)
            
    # Use a layout that spreads nodes out
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=0.8)

    # Create Plotly edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    # Create Plotly node traces
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['hover_text'])
        node_color.append(G.nodes[node]['color'])
        node_size.append(15 + np.sqrt(G.nodes[node]['size']))

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[node for node in G.nodes()], textposition="top center", hoverinfo='text', hovertext=node_text,
        marker=dict(showscale=False, color=node_color, size=node_size, line=dict(width=2, color='black')))
    
    # Use a different title for master architecture
    if genotype.lineage_id == "SYNTHESIZED_MASTER":
        title_text = f"<b>2D View: Synthesized Master Architecture</b> | Avg. Fitness: {genotype.fitness:.4f}"
    else:
        title_text = f"<b>2D View: Form {genotype.form_id}</b> | Gen {genotype.generation} | Fitness: {genotype.fitness:.4f}"

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(title=title_text, title_x=0.5, showlegend=False, hovermode='closest',
                             margin=dict(b=20, l=5, r=5, t=50), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600, plot_bgcolor='white'))
    return fig

def create_evolution_dashboard(history_df: pd.DataFrame, population: List[Genotype], evolutionary_metrics_df: pd.DataFrame) -> go.Figure:
    """Extremely advanced, comprehensive evolution analytics dashboard"""
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            '<b>Fitness Evolution by Form</b>',
            '<b>Component Score Trajectories</b>',
            '<b>Final Pareto Frontier (3D)</b>',
            '<b>Form Dominance Over Time</b>',
            '<b>Genetic Diversity (H) & Heritability (h²)</b>',
            '<b>Phenotypic Divergence (σ)</b>',
            '<b>Selection Pressure (Δ) & Mutation Rate (μ)</b>',
            '<b>Complexity & Parameter Growth</b>',
            '<b>Epigenetic Adaptation (Lamarckian Learning)</b>'
        ),
        specs=[
            [{}, {}, {'type': 'scatter3d'}],
            [{}, {'secondary_y': True}, {}],
            [{'secondary_y': True}, {'secondary_y': True}, {}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # --- Plot 1: Fitness Evolution (Enhanced) ---
    fitness_stats = history_df.groupby(['generation', 'form'])['fitness'].agg(['mean', 'std']).reset_index()
    form_names = sorted(history_df['form'].unique())
    for i, form in enumerate(form_names):
        form_data = fitness_stats[fitness_stats['form'] == form]
        if form_data.empty:
            continue

        mean_fitness = form_data['mean']
        std_fitness = form_data['std'].fillna(0)
        plot_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        
        # Add shaded area for std dev
        fig.add_trace(go.Scatter(
            x=np.concatenate([form_data['generation'], form_data['generation'][::-1]]),
            y=np.concatenate([mean_fitness + std_fitness, (mean_fitness - std_fitness)[::-1]]),
            fill='toself',
            fillcolor=plot_color,
            opacity=0.1,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=form
        ), row=1, col=1)

        # Add mean line
        fig.add_trace(go.Scatter(x=form_data['generation'], y=mean_fitness, mode='lines', name=form, legendgroup=form, line=dict(color=plot_color)), row=1, col=1)
    
    # --- Plot 2: Component Score Evolution ---
    component_scores = history_df.groupby('generation')[['accuracy', 'efficiency', 'robustness']].mean().reset_index()
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['accuracy'], name='Accuracy', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['efficiency'], name='Efficiency', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['robustness'], name='Robustness', line=dict(color='red')), row=1, col=2)

    # --- Plot 3: Pareto Front (3D) ---
    final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
    fig.add_trace(go.Scatter3d(
        x=final_gen['accuracy'], y=final_gen['efficiency'], z=final_gen['robustness'],
        mode='markers',
        marker=dict(
            size=8,
            color=final_gen['fitness'],
            colorscale='Viridis',
            colorbar=dict(title='Fitness', x=1.0, len=0.6),
            showscale=True
        ),
        text=[f"Form {int(f)}" for f in final_gen['form_id']],
        hoverinfo='text+x+y+z'
    ), row=1, col=3)
    
    # --- Plot 4: Form Dominance ---
    form_counts = history_df.groupby(['generation', 'form']).size().unstack(fill_value=0)
    form_percentages = form_counts.apply(lambda x: x / x.sum(), axis=1)
    for form in form_percentages.columns:
        fig.add_trace(go.Scatter(
            x=form_percentages.index, y=form_percentages[form],
            hoverinfo='x+y', mode='lines', name=form,
            stackgroup='one', groupnorm='percent',
            showlegend=False
        ), row=2, col=1)

    # --- Plot 5: Genetic Diversity & Heritability ---
    if not evolutionary_metrics_df.empty:
        fig.add_trace(go.Scatter(
            x=evolutionary_metrics_df['generation'], y=evolutionary_metrics_df['diversity'],
            name='Diversity (H)', line=dict(color='purple')
        ), secondary_y=False, row=2, col=2)

    heritabilities = []
    if history_df['generation'].max() > 1:
        for gen in range(1, history_df['generation'].max()):
            parent_gen = history_df[history_df['generation'] == gen - 1]
            offspring_gen = history_df[history_df['generation'] == gen]
            if len(parent_gen) > 2 and len(offspring_gen) > 2:
                h2 = EvolutionaryTheory.heritability(parent_gen['fitness'].values, offspring_gen['fitness'].values)
                heritabilities.append({'generation': gen, 'heritability': h2})
    if heritabilities:
        h2_df = pd.DataFrame(heritabilities)
        fig.add_trace(go.Scatter(
            x=h2_df['generation'], y=h2_df['heritability'],
            name='Heritability (h²)', line=dict(color='green', dash='dash')
        ), secondary_y=True, row=2, col=2)

    # --- Plot 6: Phenotypic Divergence ---
    pheno_divergence = history_df.groupby('generation')[['total_params', 'complexity']].std().reset_index()
    fig.add_trace(go.Scatter(x=pheno_divergence['generation'], y=pheno_divergence['total_params'], name='σ (Params)'), row=2, col=3)
    fig.add_trace(go.Scatter(x=pheno_divergence['generation'], y=pheno_divergence['complexity'], name='σ (Complexity)'), row=2, col=3)

    # --- Plot 7: Selection Pressure & Mutation Rate ---
    selection_diff = []
    for gen in sorted(history_df['generation'].unique()):
        gen_data = history_df[history_df['generation'] == gen]
        if len(gen_data) > 5:
            top_50_pct = gen_data.nlargest(len(gen_data) // 2, 'fitness')
            diff = top_50_pct['fitness'].mean() - gen_data['fitness'].mean()
            selection_diff.append({'generation': gen, 'selection_diff': diff})
    if selection_diff:
        sel_df = pd.DataFrame(selection_diff)
        fig.add_trace(go.Scatter(x=sel_df['generation'], y=sel_df['selection_diff'], name='Selection Δ', line=dict(color='red')), secondary_y=False, row=3, col=1)
    if not evolutionary_metrics_df.empty and 'mutation_rate' in evolutionary_metrics_df.columns:
        fig.add_trace(go.Scatter(x=evolutionary_metrics_df['generation'], y=evolutionary_metrics_df['mutation_rate'], name='Mutation Rate μ', line=dict(color='orange', dash='dash')), secondary_y=True, row=3, col=1)

    # --- Plot 8: Complexity & Parameter Growth ---
    arch_stats = history_df.groupby('generation')[['complexity', 'total_params']].mean().reset_index()
    fig.add_trace(go.Scatter(x=arch_stats['generation'], y=arch_stats['complexity'], name='Mean Complexity', line=dict(color='cyan')), secondary_y=False, row=3, col=2)
    fig.add_trace(go.Scatter(x=arch_stats['generation'], y=arch_stats['total_params'], name='Mean Params', line=dict(color='magenta', dash='dash')), secondary_y=True, row=3, col=2)

    # --- Plot 9: Epigenetic Adaptation ---
    if 'epigenetic_aptitude' in history_df.columns and history_df['epigenetic_aptitude'].abs().sum() > 1e-6:
        epigenetic_trend = history_df.groupby('generation')['epigenetic_aptitude'].mean().reset_index()
        fig.add_trace(go.Scatter(x=epigenetic_trend['generation'], y=epigenetic_trend['epigenetic_aptitude'], name='Avg. Aptitude', line=dict(color='gold')), row=3, col=3)
    else:
        fig.add_annotation(text="Epigenetics Disabled or No Adaptation", xref="paper", yref="paper", x=0.85, y=0.1, showarrow=False, row=3, col=3)

    # --- Layout and Axis Updates ---
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Fitness", row=1, col=1)
    fig.update_yaxes(title_text="Mean Score", row=1, col=2)
    fig.update_scenes(
        xaxis_title_text='Accuracy', 
        yaxis_title_text='Efficiency', 
        zaxis_title_text='Robustness',
        row=1, col=3
    )
    fig.update_yaxes(title_text="Population %", row=2, col=1)
    fig.update_yaxes(title_text="Diversity (H)", secondary_y=False, row=2, col=2)
    fig.update_yaxes(title_text="Heritability (h²)", secondary_y=True, row=2, col=2)
    fig.update_yaxes(title_text="Std. Dev (σ)", row=2, col=3)
    fig.update_yaxes(title_text="Selection Δ", secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text="Mutation Rate μ", secondary_y=True, row=3, col=1)
    fig.update_yaxes(title_text="Complexity", secondary_y=False, row=3, col=2)
    fig.update_yaxes(title_text="Parameters", secondary_y=True, row=3, col=2)
    fig.update_yaxes(title_text="Aptitude Marker", row=3, col=3)

    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text="Generation", row=i, col=j)
    fig.update_xaxes(title_text="Accuracy", row=1, col=3) # For 3D plot
    
    return fig

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="GENEVO: Elite Architecture Viewer",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # --- Password Protection ---
    def check_password():
        """Returns `True` if the user had the correct password."""

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["password"] == st.secrets["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            # First run, show input for password.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            return False
        elif not st.session_state["password_correct"]:
            # Password not correct, show input + error.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.error("Password incorrect")
            return False
        else:
            # Password correct.
            return True

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # --- Database Setup for Persistence ---
    # NOTE: You requested TinyDB, which requires an additional library.
    # To install: pip install tinydb
    db = TinyDB('genevo_db.json')
    settings_table = db.table('settings')
    results_table = db.table('results')

    # --- Load previous state if available ---
    if 'state_loaded' not in st.session_state:
        try:
            # Load settings
            saved_settings = settings_table.get(doc_id=1)
            st.session_state.settings = saved_settings if saved_settings else {}
            
            # Load results
            saved_results = results_table.get(doc_id=1)
            if saved_results:
                st.session_state.history = saved_results.get('history', [])
                st.session_state.evolutionary_metrics = saved_results.get('evolutionary_metrics', [])
                
                # Deserialize population
                pop_dicts = saved_results.get('current_population', [])
                st.session_state.current_population = [dict_to_genotype(p) for p in pop_dicts] if pop_dicts else None
                
                st.toast("Loaded previous session data.", icon="💾")
            else:
                # Initialize if no saved data
                st.session_state.history = []
                st.session_state.evolutionary_metrics = []
                st.session_state.current_population = None
        except json.JSONDecodeError:
            st.warning("⚠️ **Database Error:** Could not read the saved experiment state (`genevo_db.json`). The file may be corrupted.", icon="💾")
            st.info("This can happen if multiple sessions write to the file at once. Starting a fresh session. To fix the file, please use the **'Clear Saved State & Reset'** button in the sidebar.")
            # Initialize with empty state to allow the app to run
            st.session_state.settings = {}
            st.session_state.history = []
            st.session_state.evolutionary_metrics = []
            st.session_state.current_population = None

        st.session_state.state_loaded = True

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-top: 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧬 GENEVO: Elite Architecture Viewer</h1>', unsafe_allow_html=True)
    st.markdown('''
    <p class="sub-header">
    A focused tool to run GENEVO and immediately analyze the top-ranked evolved architectures.
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Evolution Configuration")
    
    if st.sidebar.button("⚙️ Reset to Optimal Defaults", width='stretch', help="Resets all parameters to a configuration optimized for robust and high-performance evolution.", key="reset_defaults_button"):
        # This configuration is designed for stability and achieving high accuracy.
        optimal_defaults = {
            'task_type': 'Abstract Reasoning (ARC-AGI-2)',
            'dynamic_environment': False,
            'env_change_frequency': 25,
            'num_forms': 5,
            'population_per_form': 20,
            'w_accuracy': 0.6,
            'w_efficiency': 0.15,
            'w_robustness': 0.1,
            'w_generalization': 0.15,
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'innovation_rate': 0.05,
            'enable_development': True,
            'enable_baldwin': True,
            'enable_epigenetics': True,
            'endosymbiosis_rate': 0.005,
            'epistatic_linkage_k': 2,
            'gene_flow_rate': 0.01,
            'niche_competition_factor': 1.5,
            'reintroduction_rate': 0.05,
            'max_archive_size': 100000,
            'enable_cataclysms': False,
            'cataclysm_probability': 0.01,
            'enable_red_queen': True,
            'enable_endosymbiosis': True,
            'mutation_schedule': 'Adaptive',
            'adaptive_mutation_strength': 1.0,
            'selection_pressure': 0.4,
            'enable_diversity_pressure': True,
            'diversity_weight': 0.8,
            'enable_speciation': True,
            'compatibility_threshold': 7.0,
            'num_generations': 100,
            'complexity_level': 'medium',
            'aic_pressure': 0.0,
            'fep_bias': 0.0,
            'criticality_tuning': 0.5,
            'landauer_adherence': 0.0,
            'quantum_tunneling': 0.0,
            'canalization_strength': 1.0,
            'morphogenetic_gradient': 0.0,
            'grn_plasticity': 0.0,
            'somatic_hypermutation': 0.0,
            'hgt_mobility': 0.0,
            'symbiogenesis_threshold': 0.95,
            'resource_scarcity': 1.0,
            'topological_entropy_decay': 0.05,
            'semantic_threshold': 0.0,
            'computational_temp': 1.0
        }
        st.session_state.settings = optimal_defaults
        st.toast("Parameters reset to optimal defaults!", icon="⚙️")
        st.rerun()

    # Get settings from session state, with hardcoded defaults as fallback
    s = st.session_state.get('settings', {})

    if st.sidebar.button("🗑️ Clear Saved State & Reset", width='stretch', key="clear_state_button"):
        db.truncate() # Clear all tables
        st.session_state.clear()
        st.toast("Cleared all saved data. App has been reset.", icon="🗑️")
        time.sleep(1) # Give time for toast to show
        st.rerun()

    st.sidebar.markdown("### Task Environment")
    task_options = [
        'Abstract Reasoning (ARC-AGI-2)',
        'Vision (ImageNet)',
        'Language (MMLU-Pro)',
        'Sequential Prediction',
        'Multi-Task Learning'
    ]
    default_task = s.get('task_type', 'Abstract Reasoning (ARC-AGI-2)')
    task_type = st.sidebar.selectbox(
        "Initial Task",
        task_options,
        index=task_options.index(default_task) if default_task in task_options else 0,
        help="Environmental pressure determines which architectures survive",
        key="task_type_selectbox"
    )
    
    with st.sidebar.expander("Dynamic Environment Settings"):
        dynamic_environment = st.checkbox("Enable Dynamic Environment", value=s.get('dynamic_environment', True), help="If enabled, the task will change periodically.", key="dynamic_env_checkbox")
        env_change_frequency = st.slider(
            "Change Frequency (Generations)",
            min_value=5, max_value=100, value=s.get('env_change_frequency', 25),
            help="""
            **Simulates environmental volatility.** How often the task changes.
            - **Low values:** Rapidly changing world, favors generalists.
            - **High values:** Long periods of stability, favors specialists.
            **Warning:** Very frequent changes (<10 gens) can prevent meaningful adaptation.
            """,
            disabled=not dynamic_environment,
            key="env_change_freq_slider"
        )
    
    st.sidebar.markdown("### Population Parameters")
    num_forms = st.sidebar.number_input(
        "Number of Architectural Forms",
        min_value=1, max_value=None, value=s.get('num_forms', 5), step=1,
        help="""
        **Simulates the 'Cambrian Explosion' of body plans.** The number of distinct architectural starting points.
        - **Low (1-3):** Focuses evolution on a few known good priors.
        - **High (10+):** Creates a massive, diverse ecosystem with many competing lineages.
        **WARNING:** High values create an immense total population. This is a primary factor for slow performance and potential crashes.
        """,
        key="num_forms_input"
    )
    
    population_per_form = st.sidebar.slider(
        "Population per Form", min_value=5, max_value=200, value=s.get('population_per_form', 20),
        key="pop_per_form_slider",
        help="""
        **Simulates the density of the initial gene pool.** The number of individuals for each starting architectural template.
        A larger population provides more raw genetic material and better resists random genetic drift, mimicking a vast natural population.
        **WARNING:** The 'Total Initial Population' is the key performance bottleneck. High values here will dramatically slow down each generation.
        """
    )
    
    # Add a dynamic display for total population with a warning
    total_population = num_forms * population_per_form
    st.sidebar.metric("Total Initial Population", f"{total_population:,}")
    if total_population > 300:
        st.sidebar.error(f"IMMENSE POPULATION ({total_population}). This will be extremely slow and may crash the app.")

    
    st.sidebar.markdown("### Fitness Objectives")
    with st.sidebar.expander("Multi-Objective Weights", expanded=False):
        st.info("Define the importance of each fitness objective. Weights will be normalized.")
        w_accuracy = st.slider("Accuracy Weight", 0.0, 1.0, s.get('w_accuracy', 0.6), key="w_accuracy_slider", help="**How much to value raw performance on the task.** The primary driver of optimization.")
        w_efficiency = st.slider("Efficiency Weight", 0.0, 1.0, s.get('w_efficiency', 0.15), key="w_efficiency_slider", help="**How much to penalize computational cost (parameters, connections).** Promotes smaller, faster models.")
        w_robustness = st.slider("Robustness Weight", 0.0, 1.0, s.get('w_robustness', 0.1), key="w_robustness_slider", help="**How much to value stability under perturbation.** Favors architectures that are less sensitive to noise.")
        w_generalization = st.slider("Generalization Weight", 0.0, 1.0, s.get('w_generalization', 0.15), key="w_generalization_slider", help="**How much to value traits linked to generalizing to unseen data.** Promotes modularity and plasticity.")
        
        total_w = w_accuracy + w_efficiency + w_robustness + w_generalization + 1e-9
        fitness_weights = {
            'task_accuracy': w_accuracy / total_w,
            'efficiency': w_efficiency / total_w,
            'robustness': w_robustness / total_w,
            'generalization': w_generalization / total_w
        }
        
        st.write("Normalized Weights:")
        st.json({k: f"{v:.2f}" for k, v in fitness_weights.items()})
    
    st.sidebar.markdown("### Evolutionary Operators")
    
    col1, col2 = st.sidebar.columns(2)
    mutation_rate = col1.slider(
        "Base Mutation Rate (μ)",
        min_value=0.01, max_value=0.9, value=s.get('mutation_rate', 0.2), step=0.01,
        help="""
        **The engine of variation, analogous to replication errors.** The base probability of a gene changing.
        - **Low (0.01-0.1):** Simulates a stable genome, favoring fine-tuning.
        - **High (0.5-0.9):** Simulates hypermutation or a high-stress environment, causing rapid, chaotic exploration.
        **WARNING:** Very high rates (>0.6) can lead to 'error catastrophe', where most offspring are non-viable, halting evolutionary progress.
        """,
        key="mutation_rate_slider"
    )
    crossover_rate = col2.slider(
        "Crossover Rate",
        min_value=0.0, max_value=1.0, value=s.get('crossover_rate', 0.7), step=0.05,
        help="**The engine of recombination.** The probability of creating a child from two parents. High values quickly combine successful genetic motifs from different lineages, while low values favor clonal inheritance.",
        key="crossover_rate_slider"
    )
    
    innovation_rate = st.sidebar.slider(
        "Innovation Rate (σ)",
        min_value=0.01, max_value=0.5, value=s.get('innovation_rate', 0.05), step=0.01,
        help="""
        **The engine of discovery, analogous to gene duplication and neofunctionalization.** Rate of adding new modules or connections.
        This is how architectural complexity grows over deep time.
        **WARNING:** High rates (>0.2) can lead to 'genome bloat' and chaotic, non-functional architectures, dramatically increasing computational cost without benefit.
        """,
        key="innovation_rate_slider"
    )
    
    with st.sidebar.expander("🏔️ Landscape & Speciation Control", expanded=False):
        st.markdown("Control the deep physics of the evolutionary ecosystem.")
        epistatic_linkage_k = st.slider(
            "Epistatic Linkage (K)", 0, 10, s.get('epistatic_linkage_k', 2), 1,
            help="""
            **Models the complexity of gene interactions, creating a 'rugged' fitness landscape.** From NK models, K is the number of other genes influencing a single gene's fitness.
            - **K=0:** Smooth, simple landscape (a single mountain).
            - **K > 0:** A rugged landscape with many peaks and valleys, like a mountain range.
            **WARNING:** High K (>5) creates an extremely chaotic, 'glassy' landscape that is nearly unsearchable and can prevent any meaningful adaptation.
            """,
            key="epistatic_linkage_slider"
        )
        gene_flow_rate = st.slider(
            "Gene Flow (Hybridization)", 0.0, 0.2, s.get('gene_flow_rate', 0.01), 0.005,
            help="""
            **Simulates hybridization between species.** The chance for crossover between different species, enabling major, non-linear evolutionary leaps.
            **WARNING:** This is a powerful but dangerous operator. High rates (>0.1) can destroy protected niches and cause species collapse, leading to a catastrophic loss of diversity. Use sparingly.
            """,
            disabled=not s.get('enable_speciation', True),
            key="gene_flow_rate_slider"
        )
        niche_competition_factor = st.slider(
            "Niche Competition", 0.0, 5.0, s.get('niche_competition_factor', 1.5), 0.1,
            help="""
            **Simulates the intensity of ecological competition.** How strongly individuals of the same species compete for resources (fitness sharing).
            - **>1:** Forces species to specialize to survive (niche partitioning).
            - **<1:** Allows more species to coexist.
            **WARNING:** Very high values (>3) can cause extreme specialization and may lead to species extinction if they cannot find a narrow enough niche.
            """,
            disabled=not s.get('enable_speciation', True),
            key="niche_competition_slider",
        )
        reintroduction_rate = st.slider(
            "Archive Reintroduction Rate", 0.0, 0.5, s.get('reintroduction_rate', 0.05), 0.01,
            help="""
            **Simulates a vast, ancient gene pool (the 'fossil record').** Chance to reintroduce a genotype from the archive instead of creating a new child. This is a key mechanism to combat genetic drift and simulate the vast memory of a near-infinite population.
            **WARNING:** High rates (>0.2) can slow down adaptation to the current environment by constantly reintroducing outdated or less-fit 'fossils'.
            """,
            key="reintroduction_rate_slider"
        )
        max_archive_size = st.slider(
            "Max Gene Archive Size", 1000, 1000000, s.get('max_archive_size', 100000), 5000,
            help="""
            **The memory capacity of the 'infinite' gene pool.** A larger archive provides more long-term genetic memory, preventing the permanent loss of innovations.
            **WARNING:** This directly impacts memory (RAM) usage. An immense archive (>250,000) can consume gigabytes of RAM and will crash the app on Streamlit Cloud.
            """,
            key="max_archive_size_slider"
        )
        st.info(
            "These parameters are inspired by theoretical biology to simulate complex evolutionary dynamics like epistasis and niche partitioning."
        )
    
    with st.sidebar.expander("🗂️ Experiment Management", expanded=False):
        st.markdown("Export the full configuration to reproduce this experiment, or import a previous configuration.")
        
        # Export button
        st.download_button(
            label="Export Experiment Config",
            data=json.dumps(st.session_state.settings, indent=2),
            file_name="genevo_config.json",
            mime="application/json",
            width='stretch',
            key="export_config_button"
        )

        # Import button and logic
        uploaded_file = st.file_uploader("Import Experiment Config", type="json", key="import_config_uploader")
        if uploaded_file is not None:
            new_settings = json.load(uploaded_file)
            st.session_state.settings = new_settings
            st.toast("✅ Config imported! Settings have been updated.", icon="⚙️")
            st.rerun()

    with st.sidebar.expander("🌋 Ecosystem Shocks & Dynamics", expanded=False):
        st.markdown("Introduce high-level ecosystem pressures.")
        enable_cataclysms = st.checkbox("Enable Cataclysms", value=s.get('enable_cataclysms', True), help="**Simulates mass extinctions.** Enable rare, random events like asteroid impacts (population crashes) or environmental collapses (fitness function shifts). Tests ecosystem resilience.", key="enable_cataclysms_checkbox")
        cataclysm_probability = st.slider(
            "Cataclysm Probability", 0.0, 0.5, s.get('cataclysm_probability', 0.01), 0.005,
            help="""
            **Simulates the hostility of the universe (e.g., asteroid impacts).** Per-generation chance of a cataclysmic event.
            **WARNING:** High probability (>0.1) creates an extremely unstable ecosystem where long-term, complex adaptations may be impossible to achieve before being wiped out.
            """,
            disabled=not enable_cataclysms,
            key="cataclysm_prob_slider"
        )
        enable_red_queen = st.checkbox("Enable Red Queen Dynamics", value=s.get('enable_red_queen', True), help="**'It takes all the running you can do, to keep in the same place.'** A co-evolving 'parasite' creates an arms race by targeting the most common traits, forcing continuous adaptation and preventing stagnation.", key="enable_red_queen_checkbox")
        st.info(
            "These features test the ecosystem's resilience and ability to escape static equilibria through external shocks and internal arms races."
        )


    with st.sidebar.expander("🔬 Advanced Dynamics", expanded=True):
        st.markdown("These features add deep biological complexity. You can disable them for a more classical evolutionary run.")
        enable_development = st.checkbox("Enable Developmental Program", value=s.get('enable_development', True), help="**Simulates ontogeny (growth from embryo to adult).** Allows genotypes to execute a 'developmental program' during their lifetime, such as pruning weak connections or growing modules. This separates the genotype from the final phenotype.", key="enable_development_checkbox")
        enable_baldwin = st.checkbox("Enable Baldwin Effect", value=s.get('enable_baldwin', True), help="**Models how learning shapes evolution.** An individual's `plasticity` allows it to 'learn' (improve its fitness) during its lifetime. This creates a selective pressure for architectures that are not just fit, but also good at learning (phenotypic plasticity).", key="enable_baldwin_checkbox")
        enable_epigenetics = st.checkbox("Enable Epigenetic Inheritance", value=s.get('enable_epigenetics', True), help="**Models Lamarckian-like inheritance.** Individuals pass down partially heritable 'aptitude' markers based on their life experience, allowing for very fast, non-genetic adaptation across a few generations.", key="enable_epigenetics_checkbox")
        enable_endosymbiosis = st.checkbox("Enable Endosymbiosis", value=s.get('enable_endosymbiosis', True), help="**Simulates Major Evolutionary Transitions.** A rare event where an architecture acquires a pre-evolved, successful module from another individual, simulating horizontal gene transfer and allowing for massive, instantaneous leaps in complexity.", key="enable_endosymbiosis_checkbox")
        
        endosymbiosis_rate = st.slider(
            "Endosymbiosis Rate",
            min_value=0.0, max_value=0.1, value=s.get('endosymbiosis_rate', 0.005), step=0.001,
            help="""**Rate of major evolutionary leaps.**
            **WARNING:** This should be very rare. High rates (>0.02) will lead to chaotic, Frankenstein-like architectures that are unlikely to be viable.""",
            disabled=not enable_endosymbiosis,
            key="endosymbiosis_rate_slider"
        )

        st.info(
            "Hover over the (?) on each checkbox for a detailed explanation of the dynamic."
        )

    with st.sidebar.expander("Advanced Mutation Control"):
        mutation_schedule = st.selectbox(
            "Mutation Rate Schedule",
            ['Constant', 'Linear Decay', 'Adaptive'],
            index=['Constant', 'Linear Decay', 'Adaptive'].index(s.get('mutation_schedule', 'Adaptive')),
            help="How the mutation rate changes over generations.",
            key="mutation_schedule_selectbox"
        )
        adaptive_mutation_strength = st.slider(
            "Adaptive Strength",
            min_value=0.1, max_value=5.0, value=s.get('adaptive_mutation_strength', 1.0),
            help="""
            **Simulates desperation in the face of stagnation.** A higher value causes a larger, more explosive spike in mutation rate when progress stalls, forcing the population out of a local optimum.
            **WARNING:** Very high values (>2.0) can cause wild, chaotic oscillations in mutation rate, destabilizing the evolutionary search and preventing fine-tuning.
            """,
            disabled=(mutation_schedule != 'Adaptive'),
            key="adaptive_mutation_strength_slider"
        )
    
    st.sidebar.markdown("### Selection Strategy")
    selection_pressure = st.sidebar.slider(
        "Selection Pressure", min_value=0.1, max_value=0.9, value=s.get('selection_pressure', 0.4), step=0.05,
        help="""
        **'Survival of the Fittest'.** The fraction of the population that survives to reproduce each generation.
        - **High:** Aggressive, elitist selection. Fast convergence, but high risk of getting stuck in local optima.
        - **Low:** Gentle selection. Slower progress, but maintains more diversity.
        """,
        key="selection_pressure_slider"
    )
    
    enable_diversity_pressure = st.sidebar.checkbox(
        "Enable Diversity-Pressured Selection", value=s.get('enable_diversity_pressure', True), 
        help="**Actively fights premature convergence.** Rewards novel architectures during selection, helping to prevent the population from getting stuck in an evolutionary dead-end by protecting unique but temporarily less-fit individuals."
    )
    diversity_weight = st.sidebar.slider(
        "Diversity Weight", 0.0, 5.0, s.get('diversity_weight', 0.8), 0.1,
        disabled=not enable_diversity_pressure,
        help="""
        **Controls the core Exploration vs. Exploitation trade-off.** How much to prioritize architectural novelty vs. raw fitness.
        - **>1.0:** Strongly favors exploration, creating a bizarre menagerie of unique solutions, like nature's wild diversity.
        - **<0.5:** Prioritizes exploitation, focusing on perfecting known good solutions.
        **WARNING:** Extreme values (>2.5) can cause the system to ignore fitness almost entirely, leading to beautiful but useless architectures.
        """,
        key="diversity_weight_slider"
    )
    
    with st.sidebar.expander("Speciation (NEAT-style)", expanded=True):
        enable_speciation = st.checkbox("Enable Speciation", value=s.get('enable_speciation', True), help="**Protects innovation by creating niches.** Groups similar individuals into 'species', where they only compete amongst themselves. This allows a novel but currently unoptimized idea to survive and improve without being wiped out by the dominant species.", key="enable_speciation_checkbox")
        compatibility_threshold = st.slider(
            "Compatibility Threshold",
            min_value=1.0, max_value=50.0, value=s.get('compatibility_threshold', 7.0), step=0.5,
            disabled=not enable_speciation,
            help="""
            **Defines 'what is a species'.** The maximum genomic distance for two individuals to be considered compatible.
            - **High:** Broad species definitions, leading to fewer, larger, more diverse species.
            - **Low:** Narrow definitions, leading to many small, highly specialized species.
            **WARNING:** Extreme values can disrupt speciation. Too low and every individual is its own species; too high and there is only one species.
            """,
            key="compatibility_threshold_slider"
        )
        st.info("Speciation uses a genomic distance metric based on form, module/connection differences, and parameter differences.")

    with st.sidebar.expander("♾️ Deep Evolutionary Physics & Information Dynamics", expanded=False):
        st.markdown("These parameters control the deep, underlying physics of the simulated universe, influencing information flow, developmental biology, and thermodynamic properties. **Warning:** These are highly experimental and can lead to unpredictable dynamics.")
        
        aic_pressure = st.slider(
            "Algorithmic Information Pressure (Ω)", 0.0, 1.0, s.get('aic_pressure', 0.0), 0.05,
            help="**Simulates Occam's Razor.** A selective pressure favoring genotypes with lower Kolmogorov complexity (i.e., more compressible, elegant solutions). A value of 1.0 would strongly penalize complexity.",
            key="aic_pressure_slider"
        )
        
        fep_bias = st.slider(
            "Free Energy Principle Bias (β_FEP)", 0.0, 1.0, s.get('fep_bias', 0.0), 0.05,
            help="**Models a drive for self-organization.** A bias favoring architectures structured to minimize their variational free energy (a concept from Karl Friston). This promotes models that are intrinsically good at predicting their inputs and minimizing surprise.",
            key="fep_bias_slider"
        )

        criticality_tuning = st.slider(
            "Criticality Hypothesis Tuning (σ_crit)", 0.0, 1.0, s.get('criticality_tuning', 0.5), 0.05,
            help="**Tunes the system towards the 'edge of chaos'.** A parameter that rewards architectures whose network dynamics operate in a critical state (between order and chaos), which is theorized to maximize information processing and computational power.",
            key="criticality_tuning_slider"
        )

        landauer_adherence = st.slider(
            "Landauer's Limit Adherence (L_ad)", 0.0, 1.0, s.get('landauer_adherence', 0.0), 0.05,
            help="**A penalty for thermodynamic inefficiency.** Based on Landauer's principle, this penalizes architectures for every irreversible computation (bit erasure) they perform, favoring reversible and energy-efficient designs.",
            key="landauer_adherence_slider"
        )

        quantum_tunneling = st.slider(
            "Quantum Annealing Tunneling (ħ_tunnel)", 0.0, 0.1, s.get('quantum_tunneling', 0.0), 0.005,
            help="**Simulates a quantum mechanism for escaping local optima.** A non-zero value allows genotypes to 'tunnel' through fitness barriers to adjacent, but otherwise unreachable, points in the genotype space during selection.",
            key="quantum_tunneling_slider"
        )

        st.markdown("---")

        canalization_strength = st.slider(
            "Developmental Canalization (κ)", 0.0, 5.0, s.get('canalization_strength', 1.0), 0.1,
            help="**Models developmental robustness.** How strongly the developmental process (genotype-to-phenotype mapping) resists perturbations from mutations or environmental noise. High values lead to very stable phenotypes.",
            key="canalization_strength_slider"
        )

        morphogenetic_gradient = st.slider(
            "Morphogenetic Field Gradient (∇Φ)", 0.0, 1.0, s.get('morphogenetic_gradient', 0.0), 0.05,
            help="**Controls spatial organization during development.** The strength of a simulated spatial chemical gradient that influences module placement, differentiation, and connectivity, creating body plans.",
            key="morphogenetic_gradient_slider"
        )

        grn_plasticity = st.slider(
            "Gene Regulatory Network Plasticity (γ_GRN)", 0.0, 0.1, s.get('grn_plasticity', 0.0), 0.001,
            help="**Enables meta-evolution.** The degree to which the 'rules' of evolution themselves (e.g., mutation rates, developmental rules) can evolve. A non-zero value allows the evolutionary process itself to adapt.",
            key="grn_plasticity_slider"
        )

        somatic_hypermutation = st.slider(
            "Somatic Hypermutation Rate (μ_soma)", 0.0, 0.05, s.get('somatic_hypermutation', 0.0), 0.001,
            help="**Simulates lifetime adaptation, like the immune system.** A rate of mutation that occurs *within* an individual's lifetime on specific 'plastic' modules, allowing for rapid, targeted adaptation without changing the germline DNA.",
            key="somatic_hypermutation_slider"
        )

        st.markdown("---")

        hgt_mobility = st.slider(
            "HGT Vector Mobility (ν_HGT)", 0.0, 0.01, s.get('hgt_mobility', 0.0), 0.0005,
            help="**Models extreme horizontal gene transfer.** The probability that a successful genetic module can be transferred between *unrelated* species (different Forms), enabling massive, non-linear evolutionary leaps.",
            key="hgt_mobility_slider"
        )

        symbiogenesis_threshold = st.slider(
            "Symbiogenesis Trigger (τ_sym)", 0.5, 1.0, s.get('symbiogenesis_threshold', 0.95), 0.01,
            help="**A fitness threshold for species fusion.** When two distinct species both cross this high fitness threshold, it dramatically increases the probability of a symbiogenesis (endosymbiosis) event between them.",
            key="symbiogenesis_threshold_slider"
        )

        resource_scarcity = st.slider(
            "Resource Scarcity Factor (ρ_res)", 0.5, 3.0, s.get('resource_scarcity', 1.0), 0.1,
            help="**Simulates ecosystem-wide resource limits.** A global multiplier on the 'cost' of parameters and connections. High scarcity heavily penalizes large models, forcing the entire ecosystem towards efficiency.",
            key="resource_scarcity_slider"
        )

        topological_entropy_decay = st.slider(
            "Topological Entropy Decay (λ_top)", 0.0, 0.5, s.get('topological_entropy_decay', 0.05), 0.01,
            help="**Controls diversity of network motifs.** The rate at which the diversity of local connection patterns is encouraged to decay. High decay forces the system to settle on a few proven motifs; low decay encourages exploration of new ones.",
            key="topological_entropy_decay_slider"
        )

        semantic_threshold = st.slider(
            "Semantic Information Threshold (I(X;Y))", 0.0, 0.5, s.get('semantic_threshold', 0.0), 0.01,
            help="**Enforces meaningful complexity.** A minimum threshold of mutual information between an architecture's structural properties and its performance. Penalizes 'complex for nothing' architectures where structure is uncorrelated with function.",
            key="semantic_threshold_slider"
        )

        computational_temp = st.slider(
            "Computational Temperature (T_comp)", 0.0, 2.0, s.get('computational_temp', 1.0), 0.05,
            help="**Noise in the genotype-phenotype map.** Analogous to thermodynamic temperature, this controls the stochasticity of the developmental process. High temperature leads to more variable and noisy phenotypic expression for the same genotype.",
            key="computational_temp_slider"
        )

    st.sidebar.markdown("### Experiment Settings")
    num_generations = st.sidebar.slider(
        "Generations",
        min_value=10, max_value=1000, value=s.get('num_generations', 100),
        help="""
        **Simulates deep evolutionary time.** The number of cycles of selection and reproduction.
        - **Low (10-50):** Short-term adaptation.
        - **High (200+):** Allows for major evolutionary transitions and complex features to emerge.
        **WARNING:** Immense timescales (>200) will take a very long time to compute. A 1000-generation run could take hours.
        """,
        key="num_generations_slider"
    )
    
    complexity_options = ['minimal', 'medium', 'high']
    complexity_level = st.sidebar.select_slider(
        "Initial Complexity",
        options=complexity_options,
        value=s.get('complexity_level', 'medium'),
        key="complexity_level_select_slider"
    )
    
    # --- Collect and save current settings ---
    current_settings = {
        'task_type': task_type,
        'dynamic_environment': dynamic_environment,
        'env_change_frequency': env_change_frequency,
        'num_forms': num_forms,
        'population_per_form': population_per_form,
        'w_accuracy': w_accuracy,
        'w_efficiency': w_efficiency,
        'w_robustness': w_robustness,
        'w_generalization': w_generalization,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'innovation_rate': innovation_rate,
        'enable_development': enable_development,
        'enable_baldwin': enable_baldwin,
        'enable_epigenetics': enable_epigenetics,
        'endosymbiosis_rate': endosymbiosis_rate,
        'epistatic_linkage_k': epistatic_linkage_k,
        'gene_flow_rate': gene_flow_rate,
        'niche_competition_factor': niche_competition_factor,
        'max_archive_size': max_archive_size,
        'reintroduction_rate': reintroduction_rate,
        'enable_cataclysms': enable_cataclysms,
        'cataclysm_probability': cataclysm_probability,
        'enable_red_queen': enable_red_queen,
        'enable_endosymbiosis': enable_endosymbiosis,
        'mutation_schedule': mutation_schedule,
        'adaptive_mutation_strength': adaptive_mutation_strength,
        'selection_pressure': selection_pressure,
        'enable_speciation': enable_speciation,
        'enable_diversity_pressure': enable_diversity_pressure,
        'diversity_weight': diversity_weight,
        'compatibility_threshold': compatibility_threshold,
        'num_generations': num_generations,
        'complexity_level': complexity_level,
        'aic_pressure': aic_pressure,
        'fep_bias': fep_bias,
        'criticality_tuning': criticality_tuning,
        'landauer_adherence': landauer_adherence,
        'quantum_tunneling': quantum_tunneling,
        'canalization_strength': canalization_strength,
        'morphogenetic_gradient': morphogenetic_gradient,
        'grn_plasticity': grn_plasticity,
        'somatic_hypermutation': somatic_hypermutation,
        'hgt_mobility': hgt_mobility,
        'symbiogenesis_threshold': symbiogenesis_threshold,
        'resource_scarcity': resource_scarcity,
        'topological_entropy_decay': topological_entropy_decay,
        'semantic_threshold': semantic_threshold,
        'computational_temp': computational_temp
    }
    
    # Save settings to DB if they have changed
    if current_settings != st.session_state.settings:
        st.session_state.settings = current_settings
        if settings_table.get(doc_id=1):
            settings_table.update(current_settings, doc_ids=[1])
        else:
            settings_table.insert(current_settings)
        st.toast("Settings saved.", icon="⚙️")

    st.sidebar.markdown("---")
    
    # Run evolution button
    if st.sidebar.button("⚡ Initiate Evolution", type="primary", width='stretch', key="initiate_evolution_button"):
        st.session_state.history = []
        st.session_state.evolutionary_metrics = [] # type: ignore
        st.session_state.gene_archive = [] # Initialize the infinite gene pool
        
        # Initialize population
        population = []
        for form_id in range(1, num_forms + 1):
            for _ in range(population_per_form):
                genotype = initialize_genotype(form_id, complexity_level)
                genotype.generation = 0
                population.append(genotype)
                st.session_state.gene_archive.append(genotype.copy()) # Seed the archive
        
        # For adaptive mutation
        last_best_fitness = -1
        stagnation_counter = 0
        current_mutation_rate = mutation_rate
        
        # For dynamic environment
        current_task = task_type
        
        # For ecosystem dynamics
        st.session_state.cataclysm_recovery_mode = 0
        st.session_state.cataclysm_weights = None
        # The parasite now targets the most common activation function across all module types.
        st.session_state.parasite_profile = {'target_activation': random.choice(['relu', 'gelu', 'silu', 'swish'])}

        # Progress tracking
        progress_container = st.empty()
        metrics_container = st.empty()
        status_text = st.empty()
        
        # Evolution loop
        for gen in range(num_generations):
            # --- Ecosystem Dynamics ---
            active_fitness_weights = fitness_weights
            
            # Cataclysm Recovery
            if st.session_state.cataclysm_recovery_mode > 0:
                st.session_state.cataclysm_recovery_mode -= 1
                active_fitness_weights = st.session_state.cataclysm_weights
                if st.session_state.cataclysm_recovery_mode == 0:
                    st.toast("🌍 Environmental pressures have normalized.", icon="✅")
                    st.session_state.cataclysm_weights = None

            # Cataclysm Trigger
            elif enable_cataclysms and random.random() < cataclysm_probability:
                event_type = random.choice(['extinction', 'collapse'])
                if event_type == 'extinction' and len(population) > 10:
                    st.warning(f"💥 Mass Extinction Event! A genetic bottleneck has occurred.")
                    st.toast("💥 Mass Extinction!", icon="☄️")
                    survivor_count = max(5, int(len(population) * 0.1))
                    population = random.sample(population, k=survivor_count)
                elif event_type == 'collapse':
                    st.warning(f"📉 Environmental Collapse! Fitness objectives have drastically shifted.")
                    st.toast("📉 Environmental Collapse!", icon="🌪️")
                    st.session_state.cataclysm_recovery_mode = 5 # Lasts for 5 generations
                    collapse_target = random.choice(list(fitness_weights.keys()))
                    st.session_state.cataclysm_weights = {k: 0.05 for k in fitness_weights}
                    st.session_state.cataclysm_weights[collapse_target] = 0.8
                    active_fitness_weights = st.session_state.cataclysm_weights

            # Red Queen Parasite Info
            parasite_display = status_text.empty()

            # Handle dynamic environment
            if dynamic_environment and gen > 0 and gen % env_change_frequency == 0:
                previous_task = current_task
                current_task = random.choice([t for t in task_options if t != previous_task])
                st.toast(f"🌍 Environment Shift! New Task: {current_task}", icon="🔄")
                time.sleep(1.0)
            
            status_text.markdown(f"### 🧬 Generation {gen + 1}/{num_generations} | Task: **{current_task}**")
            
            # --- Apply developmental rules ---
            # This simulates lifetime development like pruning and growth before evaluation
            if enable_development:
                for i in range(len(population)):
                    population[i] = apply_developmental_rules(population[i], stagnation_counter)

            # Evaluate fitness
            all_scores = []
            for individual in population:
                # Pass the weights from the sidebar
                # Pass the flags for advanced dynamics
                fitness, component_scores = evaluate_fitness(individual, current_task, gen, active_fitness_weights, enable_epigenetics, enable_baldwin, epistatic_linkage_k, st.session_state.parasite_profile if enable_red_queen else None)
                individual.fitness = fitness
                individual.generation = gen
                individual.age += 1
                all_scores.append(component_scores)
            
            # Record detailed history
            for individual, scores in zip(population, all_scores):
                st.session_state.history.append({
                    'generation': gen,
                    'form': f'Form {individual.form_id}',
                    'form_id': individual.form_id,
                    'fitness': individual.fitness,
                    'accuracy': scores['task_accuracy'],
                    'efficiency': scores['efficiency'],
                    'robustness': scores['robustness'],
                    'generalization': scores['generalization'],
                    'total_params': sum(m.size for m in individual.modules),
                    'num_connections': len(individual.connections),
                    'complexity': individual.complexity,
                    'lineage_id': individual.lineage_id,
                    'parent_ids': individual.parent_ids
                })
            
            # Compute evolutionary metrics
            fitness_array = np.array([ind.fitness for ind in population])
            diversity = EvolutionaryTheory.genetic_diversity(population)
            fisher_info = EvolutionaryTheory.fisher_information(population, fitness_array)

            # Display real-time metrics
            with metrics_container.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Best Fitness", f"{fitness_array.max():.4f}")
                col2.metric("Mean Fitness", f"{fitness_array.mean():.4f}")
                col3.metric("Diversity (H)", f"{diversity:.3f}")
                col4.metric("Mutation Rate (μ)", f"{current_mutation_rate:.3f}")
                # Placeholder for species count, will be updated below
                if enable_red_queen:
                    target_act = st.session_state.parasite_profile.get('target_activation', 'N/A')
                    parasite_display.info(f"**Red Queen Active:** Parasite targeting `{target_act}` activation function.")
                else:
                    parasite_display.empty()
                species_metric = col5.metric("Species Count", "N/A")

            st.session_state.evolutionary_metrics.append({
                'generation': gen,
                'diversity': diversity,
                'fisher_info': fisher_info,
                'best_fitness': fitness_array.max(),
                'mean_fitness': fitness_array.mean()
            })
            
            # Selection
            if enable_speciation:
                species = []
                for ind in population:
                    found_species = False
                    for s in species:
                        representative = s['representative']
                        dist = genomic_distance(ind, representative)
                        if dist < compatibility_threshold:
                            s['members'].append(ind)
                            found_species = True
                            break
                    if not found_species:
                        species.append({'representative': ind, 'members': [ind]})
                
                species_metric.metric("Species Count", f"{len(species)}")
                
                # Apply fitness sharing
                # Also calculate novelty if diversity pressure is on
                for s in species:
                    species_size = len(s['members'])
                    if species_size > 0:
                        for member in s['members']:
                            member.adjusted_fitness = member.fitness / (species_size ** niche_competition_factor)
                
                if enable_diversity_pressure:
                    # Calculate novelty for each individual across the whole population
                    for ind in population:
                        distances = [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id]
                        distances = [d for d in distances if d != float('inf')]
                        if distances:
                            k = min(10, len(distances))
                            distances.sort()
                            ind.novelty_score = np.mean(distances[:k])
                        else:
                            ind.novelty_score = 0.0
                    
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    if max_novelty > 0:
                        for ind in population:
                            ind.selection_score = ind.adjusted_fitness + diversity_weight * (ind.novelty_score / max_novelty)
                    else:
                        for ind in population:
                            ind.selection_score = ind.adjusted_fitness
                    
                    selection_key = lambda x: x.selection_score
                else:
                    selection_key = lambda x: x.adjusted_fitness

            else:
                species_metric.metric("Species Count", f"{len(set(ind.form_id for ind in population))}")
                if enable_diversity_pressure:
                    for ind in population:
                        distances = [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id]
                        distances = [d for d in distances if d != float('inf')]
                        if distances:
                            k = min(10, len(distances))
                            distances.sort()
                            ind.novelty_score = np.mean(distances[:k])
                        else:
                            ind.novelty_score = 0.0
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    for ind in population:
                        ind.selection_score = ind.fitness + diversity_weight * (ind.novelty_score / (max_novelty + 1e-9))
                    selection_key = lambda x: x.selection_score
                else:
                    selection_key = lambda x: x.fitness

            population.sort(key=selection_key, reverse=True)

            num_survivors = max(2, int(len(population) * selection_pressure))
            survivors = population[:num_survivors]
            
            # Calculate selection differential
            selected_idx = np.arange(num_survivors)
            sel_diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)

            # Red Queen Parasite Evolution
            if enable_red_queen and survivors:
                trait_counts = Counter()
                for ind in survivors:
                    for module in ind.modules:
                        trait_counts[module.activation] += 1
                if trait_counts:
                    st.session_state.parasite_profile['target_activation'] = trait_counts.most_common(1)[0][0]
            
            # Reproduction
            offspring = []
            while len(offspring) < len(population) - len(survivors):
                if random.random() < reintroduction_rate and st.session_state.gene_archive:
                    # Reintroduce a "fossil" from the infinite gene pool
                    child = random.choice(st.session_state.gene_archive).copy()
                    child = mutate(child, current_mutation_rate * 1.5, innovation_rate * 1.5) # Mutate it heavily to adapt it
                    child.generation = gen + 1
                    offspring.append(child)
                else:
                    # --- Create one viable child via normal reproduction, with retries ---
                    max_attempts = 20
                    for _ in range(max_attempts):
                        # Tournament selection using the appropriate fitness key
                        parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                        
                        if random.random() < crossover_rate:
                            if enable_speciation and random.random() < gene_flow_rate and len(survivors) > 1:
                                # Gene Flow: select any other survivor, ignoring species
                                parent2_candidates = [s for s in survivors if s.lineage_id != parent1.lineage_id]
                                parent2 = random.choice(parent2_candidates) if parent2_candidates else parent1
                            elif len(survivors) > 1:
                                # Normal Crossover: select compatible parent
                                compatible = [s for s in survivors if s.form_id == parent1.form_id and s.lineage_id != parent1.lineage_id]
                                parent2 = max(random.sample(compatible, min(2, len(compatible))), key=selection_key) if compatible else parent1 # type: ignore
                            else:
                                parent2 = parent1
                            child = crossover(parent1, parent2, crossover_rate)
                        else:
                            child = parent1.copy()
                        
                        # Mutation and other operators
                        child = mutate(child, current_mutation_rate, innovation_rate)
                        if enable_endosymbiosis and random.random() < endosymbiosis_rate and survivors:
                            child = apply_endosymbiosis(child, survivors)
                        
                        # Viability Selection: Ensure the child is a functional network
                        if is_viable(child):
                            child.generation = gen + 1
                            offspring.append(child)
                            st.session_state.gene_archive.append(child.copy()) # Add new viable child to archive
                            # Prune archive if it exceeds max size to manage memory
                            max_size = st.session_state.settings.get('max_archive_size', 10000)
                            if len(st.session_state.gene_archive) > max_size:
                                st.session_state.gene_archive = random.sample(st.session_state.gene_archive, max_size)
                            break # Found a viable child, move to next offspring
                    else: # for-else: runs if the loop finished without break
                        # Fallback if no viable child was found after many attempts
                        parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                        child = mutate(parent1.copy(), current_mutation_rate, innovation_rate)
                        child.generation = gen + 1
                        offspring.append(child)
            
            # Clean up temporary attribute
            if enable_speciation:
                for ind in population: # Clean up all temporary scores
                    for attr in ['adjusted_fitness', 'novelty_score', 'selection_score']: # type: ignore
                        if hasattr(ind, attr): delattr(ind, attr)

            # Update mutation rate for next generation
            if mutation_schedule == 'Linear Decay':
                current_mutation_rate = mutation_rate * (1.0 - ((gen + 1) / num_generations))
            elif mutation_schedule == 'Adaptive':
                current_best_fitness = fitness_array.max()
                if current_best_fitness > last_best_fitness:
                    stagnation_counter = 0
                    current_mutation_rate = max(0.05, current_mutation_rate * 0.95) # Anneal
                else:
                    stagnation_counter += 1
                
                if stagnation_counter > 3: # If stagnated for >3 generations
                    current_mutation_rate = min(0.8, current_mutation_rate * (1 + adaptive_mutation_strength)) # Spike
                
                last_best_fitness = current_best_fitness

            population = survivors + offspring
            
            # Update progress
            progress_container.progress((gen + 1) / num_generations)
        
        st.session_state.current_population = population
        
        # --- Save results to DB ---
        serializable_population = [genotype_to_dict(p) for p in population]
        
        results_to_save = {
            'history': st.session_state.history,
            'evolutionary_metrics': st.session_state.evolutionary_metrics,
            'current_population': serializable_population
        }
        if results_table.get(doc_id=1):
            results_table.update(results_to_save, doc_ids=[1])
        else:
            results_table.insert(results_to_save)
        
        status_text.markdown("### ✅ Evolution Complete! Results saved.")
    
    # Display results
    if st.session_state.history and st.session_state.current_population:
        st.markdown("---")
        
        # --- Setup for deep analysis ---
        s = st.session_state.settings
        task_type = s.get('task_type', 'Abstract Reasoning (ARC-AGI-2)')
        enable_epigenetics = s.get('enable_epigenetics', True)
        enable_baldwin = s.get('enable_baldwin', True)
        epistatic_linkage_k = s.get('epistatic_linkage_k', 0)
        
        w_accuracy = s.get('w_accuracy', 0.5)
        w_efficiency = s.get('w_efficiency', 0.2)
        w_robustness = s.get('w_robustness', 0.1)
        w_generalization = s.get('w_generalization', 0.2)
        total_w = w_accuracy + w_efficiency + w_robustness + w_generalization + 1e-9
        fitness_weights = {
            'task_accuracy': w_accuracy / total_w,
            'efficiency': w_efficiency / total_w,
            'robustness': w_robustness / total_w,
            'generalization': w_generalization / total_w
        }
        eval_params = {
            'enable_epigenetics': enable_epigenetics,
            'enable_baldwin': enable_baldwin,
            'epistatic_linkage_k': epistatic_linkage_k
        }

        # Best evolved architectures
        st.header("🏛️ Elite Evolved Architectures: A Deep Dive")
        st.markdown("This section provides a multi-faceted analysis of the top-performing genotypes from the final population, deconstructing the architectural, causal, and evolutionary properties that contributed to their success.")
        
        population = st.session_state.current_population
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Show top 3
        for i, individual in enumerate(population[:3]):
            expander_title = f"**Rank {i+1}:** Form `{individual.form_id}` | Lineage `{individual.lineage_id}` | Fitness: `{individual.fitness:.4f}`"
            with st.expander(expander_title, expanded=(i==0)):
                
                # Define tabs for the deep dive
                tab_vitals, tab_causal, tab_evo, tab_ancestry, tab_code = st.tabs([
                    "🌐 Vitals & Architecture", 
                    "🔬 Causal & Structural Analysis", 
                    "🧬 Evolutionary & Developmental Potential",
                    "🌳 Genealogy & Ancestry",
                    "💻 Code Export"
                ])
                # --- TAB 1: Vitals & Architecture ---
                with tab_vitals:
                    vitals_col1, vitals_col2 = st.columns([1, 1])
                    with vitals_col1:
                        st.markdown("#### Quantitative Profile")
                        st.metric("Fitness", f"{individual.fitness:.4f}")
                        st.metric("Task Accuracy (Sim.)", f"{individual.accuracy:.3f}")
                        st.metric("Efficiency Score", f"{individual.efficiency:.3f}")
                        st.metric("Robustness Score", f"{individual.robustness:.3f}")
                        st.metric("Architectural Complexity", f"{individual.complexity:.3f}")
                        st.metric("Age", f"{individual.age} generations")

                    with vitals_col2:
                        st.markdown("#### Architectural Blueprint")
                        st.write(f"**Total Parameters:** `{sum(m.size for m in individual.modules):,}`")
                        st.write(f"**Modules:** `{len(individual.modules)}` | **Connections:** `{len(individual.connections)}`")
                        st.write(f"**Parent(s):** `{', '.join(individual.parent_ids)}`")
                        
                        st.markdown("###### Module Composition:")
                        module_counts = Counter(m.module_type for m in individual.modules)
                        for mtype, count in module_counts.items():
                            st.write(f"- `{count}` x **{mtype.capitalize()}**")
                        
                        st.markdown("###### Meta-Parameters:")
                        st.json({k: f"{v:.4f}" for k, v in individual.meta_parameters.items()})

                st.markdown("---")
                st.markdown("#### Visualizations")
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.markdown("###### 3D Interactive View")
                    st.plotly_chart(
                        visualize_genotype_3d(individual),
                        width='stretch',
                        key=f"elite_3d_{i}_{individual.lineage_id}"
                    )
                with vis_col2:
                    st.markdown("###### 2D Static View")
                    st.plotly_chart(
                        visualize_genotype_2d(individual),
                        width='stretch',
                        key=f"elite_2d_{i}_{individual.lineage_id}"
                    )

                # --- TAB 2: Causal & Structural Analysis ---
                with tab_causal:
                    st.markdown("This tab dissects the functional importance of the architecture's components.")
                    causal_col1, causal_col2 = st.columns(2)

                    with causal_col1:
                        st.subheader("Lesion Sensitivity Analysis")
                        st.markdown("We computationally 'remove' each component and measure the drop in fitness. A larger drop indicates a more **critical** component.")
                        
                        with st.spinner(f"Performing lesion analysis for Rank {i+1}..."):
                            criticality_scores = analyze_lesion_sensitivity(
                                individual, individual.fitness, task_type, fitness_weights, eval_params
                            )
                        
                        sorted_criticality = sorted(criticality_scores.items(), key=lambda item: item[1], reverse=True)
                        
                        st.markdown("###### Most Critical Components:")
                        for j, (component, score) in enumerate(sorted_criticality[:5]):
                            st.metric(
                                label=f"#{j+1} {component}",
                                value=f"{score:.4f} Fitness Drop",
                                help="The reduction in overall fitness when this component is removed."
                            )

                    with causal_col2:
                        st.subheader("Information Flow Backbone")
                        st.markdown("This analysis identifies the **causal backbone**—the key modules that act as bridges for information flow, identified via `betweenness centrality`.")
                        
                        with st.spinner(f"Analyzing information flow for Rank {i+1}..."):
                            centrality_scores = analyze_information_flow(individual)
                        
                        sorted_centrality = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)
                        st.markdown("###### Causal Backbone Nodes:")
                        for j, (module_id, score) in enumerate(sorted_centrality[:5]):
                            st.metric(label=f"#{j+1} Module: {module_id}", value=f"{score:.3f} Centrality", help="A normalized score of how critical this node is for information routing.")

                    st.markdown("---")
                    st.subheader("Genetic Load & Neutrality")
                    st.markdown("This analysis identifies **neutral components** ('junk DNA') with little effect when removed, and calculates the **genetic load**—the fitness cost of slightly harmful, non-lethal components.")
                    with st.spinner(f"Calculating genetic load for Rank {i+1}..."):
                        load_data = analyze_genetic_load(criticality_scores)

                    load_col1, load_col2 = st.columns(2)
                    load_col1.metric("Neutral Component Count", f"{load_data['neutral_component_count']}", help="Number of modules/connections with near-zero impact when lesioned.")
                    load_col2.metric("Genetic Load", f"{load_data['genetic_load']:.4f}", help="Total fitness reduction from slightly deleterious, non-critical components.")

                # --- TAB 3: Evolutionary & Developmental Potential ---
                with tab_evo:
                    st.markdown("This tab probes the genotype's potential for future adaptation and its programmed developmental trajectory.")
                    evo_col1, evo_col2 = st.columns(2)

                    with evo_col1:
                        st.subheader("Evolvability vs. Robustness")
                        st.markdown("We generate 50 mutants to measure the trade-off between **robustness** (resisting negative mutations) and **evolvability** (producing beneficial mutations).")
                        
                        with st.spinner(f"Analyzing mutational landscape for Rank {i+1}..."):
                            evo_robust_data = analyze_evolvability_robustness(
                                individual, task_type, fitness_weights, eval_params
                            )
                        
                        st.metric("Robustness Score", f"{evo_robust_data['robustness']:.4f}", help="Average fitness loss from deleterious mutations. Higher = more robust.")
                        st.metric("Evolvability Score", f"{evo_robust_data['evolvability']:.4f}", help="Maximum fitness gain from a single mutation.")

                        dist_df = pd.DataFrame(evo_robust_data['distribution'], columns=['Fitness Change'])
                        fig = px.histogram(dist_df, x="Fitness Change", nbins=20, title="Distribution of Mutational Effects")
                        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                        st.plotly_chart(fig, width='stretch', key=f"mutational_effects_hist_{i}_{individual.lineage_id}")

                    with evo_col2:
                        st.subheader("Developmental Trajectory")
                        st.markdown("This simulates the genotype's 'lifetime,' showing how its developmental program (pruning, proliferation) alters its structure over time.")
                        
                        with st.spinner(f"Simulating development for Rank {i+1}..."):
                            dev_traj_df = analyze_developmental_trajectory(individual)
                        
                        fig = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), showlegend=False)
                        st.plotly_chart(fig, width='stretch', key=f"dev_trajectory_line_{i}_{individual.lineage_id}")

                # --- TAB 4: Genealogy & Ancestry ---
                with tab_ancestry:
                    st.markdown("This tab traces the lineage of the genotype, comparing it to its direct ancestors to understand the evolutionary step that led to its success.")
                    parent_ids = individual.parent_ids
                    st.subheader(f"Direct Ancestors: `{', '.join(parent_ids)}`")

                    parents = [p for p in population if p.lineage_id in parent_ids]

                    if not parents:
                        st.info("Parents of this genotype are not in the final population (they were not selected in a previous generation).")
                    else:
                        parent_cols = st.columns(len(parents))
                        for k, parent in enumerate(parents):
                            with parent_cols[k]:
                                st.markdown(f"##### Parent: `{parent.lineage_id}`")
                                st.markdown(f"**Form:** `{parent.form_id}` | **Fitness:** `{parent.fitness:.4f}`")
                                
                                # Genomic distance
                                distance = genomic_distance(individual, parent)
                                st.metric("Genomic Distance to Child", f"{distance:.3f}")

                                # Compare key stats
                                st.markdown("###### Evolutionary Step:")
                                param_delta = sum(m.size for m in individual.modules) - sum(m.size for m in parent.modules)
                                st.metric("Parameter Change", f"{param_delta:+,}", delta_color="off")
                                
                                complexity_delta = individual.complexity - parent.complexity
                                st.metric("Complexity Change", f"{complexity_delta:+.3f}", delta_color="off")

                                # Visualize parent
                                st.plotly_chart(
                                    visualize_genotype_2d(parent),
                                    width='stretch',
                                    key=f"parent_2d_{i}_{k}_{parent.lineage_id}_for_{individual.lineage_id}"
                                )

                # --- TAB 5: Code Export ---
                with tab_code:
                    st.markdown("The genotype can be translated into functional code for deep learning frameworks, providing a direct path from discovery to application.")
                    
                    code_col1, code_col2 = st.columns(2)
                    with code_col1:
                        st.subheader("PyTorch Code")
                        pytorch_code = generate_pytorch_code(individual)
                        st.code(pytorch_code, language='python')
                    
                    with code_col2:
                        st.subheader("TensorFlow / Keras Code")
                        tensorflow_code = generate_tensorflow_code(individual)
                        st.code(tensorflow_code, language='python')
        
        # --- Download all data ---
        st.markdown("---")
        st.subheader("Download Experiment Data")
        st.markdown("Download all experiment data, including settings, history, evolutionary metrics, and the final population, for offline analysis and reproducibility.")

        # Prepare data for download
        all_data = {
            'settings': st.session_state.settings,
            'history': st.session_state.history,
            'evolutionary_metrics': st.session_state.evolutionary_metrics,
            'final_population': [genotype_to_dict(p) for p in st.session_state.current_population] if st.session_state.current_population else []
        }
        all_data_json = json.dumps(all_data, indent=2)

        # Create download button
        st.download_button(
            label="Download All Experiment Data (JSON)",
            data=all_data_json,
            file_name="genevo_experiment_data.json",
            mime="application/json",
            key="download_all_data_button"
        )

    st.sidebar.markdown("---")

    with st.sidebar.expander("✨Understanding Your Results: A Guide to Evolutionary Dynamics", expanded=False):
        st.markdown("""
        Have you noticed that your results are incredibly diverse? Sometimes the top-ranked architectures are nearly identical, while other times they are wildly different. **This is not a bug—it is the strongest possible evidence that you are witnessing true, complex evolutionary dynamics at work!** 🔬

        This system is not a simple optimizer; it's a digital ecosystem. The variety in your outcomes is driven by a fascinating tug-of-war between powerful evolutionary forces. Here’s what’s happening:
        """)

        st.markdown("---")

        st.markdown("""
        #### 🎯 The Force of Exploitation (Convergent Evolution)
        *When your top 3 ranks are almost 100% similar...*

        This is **Convergent Evolution**. The algorithm has found a highly successful architectural design early in the run. Like a team of mountaineers finding a very promising path up a huge mountain, the system decides to focus all its energy on that single path.

        - **What's happening:** Instead of searching for new mountains, evolution is aggressively *exploiting* this known good solution. The top-ranked individuals are like siblings or close cousins, each a minor refinement of the same dominant ancestor.
        - **Your Controls:** This behavior is promoted by a high **Selection Pressure** and a low **Diversity Weight**. You are telling the system: "Find the best solution and perfect it, no matter what."
        """)

        st.markdown("---")

        st.markdown("""
        #### 🗺️ The Force of Exploration (Divergent Evolution)
        *When your top 3 ranks are totally different and unique...*

        This is **Divergent Evolution**. The system is actively rewarding novelty and searching different corners of the vast "solution space." It's like sending scouts in all directions across a massive mountain range. Each scout finds a completely different peak and starts climbing.

        - **What's happening:** The final winners are the champions of entirely different strategies. One might be a small, efficient network, while another is a large, powerful one. The system is *exploring* multiple, distinct solutions simultaneously.
        - **Your Controls:** This is driven by a high **Diversity Weight** and a "rugged" fitness landscape (created by setting **Epistatic Linkage (K) > 0**). You are telling the system: "Don't just give me one good answer; find me a portfolio of different, creative solutions."
        """)

        st.markdown("---")

        st.markdown("""
        #### 🏞️ The Hybrid Outcome (Niche Formation)
        *When two ranks are similar, but one is unique...*

        This is a beautiful and realistic hybrid scenario. It's a sign of **Niche Formation**. The main group of climbers has focused on the highest, most obvious peak. However, the system has kept a smaller, specialized team alive because they found a different, slightly lower peak that has unique advantages (e.g., it's more "efficient" or "robust").

        - **What's happening:** The system has found a dominant strategy (the two similar architectures) but was forced by diversity pressure to protect a second, viable strategy. It has successfully carved out two different ecological niches.
        - **Your Controls:** This emerges from a fine balance between **Selection Pressure** and **Diversity Weight**.
        """)

        st.markdown("---")

        st.markdown("""
        #### 🦋 The Power of Randomness (The "Butterfly Effect")
        *Why every run is unique, even with the same settings...*

        This is **Historical Contingency**. In evolution, tiny, random events at the beginning can lead to massive, unpredictable differences later on. A single "lucky" mutation in generation 2 can set a lineage on a path to greatness that was completely missed in another run.

        - **What's happening:** The stochastic (random) nature of mutation and crossover means the evolutionary path is never predetermined. This is what makes evolution a genuinely creative process, not just a deterministic search.
        - **Your Controls:** This is an inherent property of the algorithm! By clicking "Initiate Evolution," you are rolling the dice and starting a new, unique history for your digital universe.

        **Conclusion:** You are not just running a program; you are an experimental scientist setting the laws of physics for a new universe. The diverse results are the proof that your universe is alive with complex, emergent behavior. **Embrace the variety!**
        """)

    st.sidebar.info(
        "**GENEVO** is a research prototype demonstrating advanced concepts in neuroevolution. "
        "Architectures are simulated and not trained on real data."
    )


if __name__ == "__main__":
    main()
