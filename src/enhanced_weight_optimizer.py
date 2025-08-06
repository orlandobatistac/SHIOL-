
"""
SHIOL+ Enhanced Weight Optimizer
================================

Sistema avanzado de optimización de pesos que utiliza múltiples algoritmos
y técnicas de machine learning para encontrar la configuración óptima.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import differential_evolution, minimize, basinhopping
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.database import get_performance_analytics, get_all_draws
from src.intelligent_generator import DeterministicGenerator, PlayScorer


class EnhancedWeightOptimizer:
    """
    Optimizador avanzado de pesos que utiliza múltiples algoritmos
    y técnicas de aprendizaje para encontrar la configuración óptima.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.scorer = PlayScorer(historical_data)
        
        # Configuración de optimización
        self.weight_bounds = {
            'probability': (0.2, 0.6),     # 20-60%
            'diversity': (0.1, 0.4),       # 10-40%
            'historical': (0.1, 0.4),      # 10-40%
            'risk_adjusted': (0.05, 0.3)   # 5-30%
        }
        
        # Algoritmos disponibles
        self.algorithms = {
            'differential_evolution': self._optimize_differential_evolution,
            'bayesian_optimization': self._optimize_bayesian,
            'random_search': self._optimize_random_search,
            'gradient_descent': self._optimize_gradient_descent,
            'genetic_algorithm': self._optimize_genetic_algorithm,
            'ensemble': self._optimize_ensemble
        }
        
        logger.info("EnhancedWeightOptimizer initialized with advanced algorithms")
    
    def optimize_weights_advanced(self, current_weights: Dict[str, float],
                                performance_data: Dict[str, Any],
                                algorithm: str = 'ensemble',
                                max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimiza pesos usando algoritmos avanzados.
        
        Args:
            current_weights: Pesos actuales
            performance_data: Datos de performance histórica
            algorithm: Algoritmo a utilizar
            max_iterations: Máximo número de iteraciones
            
        Returns:
            Dict con resultados de optimización
        """
        logger.info(f"Starting advanced weight optimization with {algorithm} algorithm...")
        
        optimization_results = {
            'optimization_timestamp': datetime.now().isoformat(),
            'algorithm_used': algorithm,
            'current_weights': current_weights.copy(),
            'optimized_weights': None,
            'performance_improvement': 0.0,
            'optimization_history': [],
            'convergence_achieved': False,
            'total_iterations': 0,
            'execution_time_seconds': 0,
            'confidence_score': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Preparar datos para optimización
            optimization_data = self._prepare_optimization_data(performance_data)
            
            if optimization_data is None:
                logger.warning("Insufficient data for weight optimization")
                return optimization_results
            
            # Ejecutar algoritmo seleccionado
            if algorithm in self.algorithms:
                optimizer_func = self.algorithms[algorithm]
                optimized_weights, opt_history = optimizer_func(
                    current_weights, optimization_data, max_iterations
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Validar pesos optimizados
            if optimized_weights is not None:
                optimized_weights = self._validate_and_normalize_weights(optimized_weights)
                
                # Evaluar mejora
                current_score = self._evaluate_weights(current_weights, optimization_data)
                optimized_score = self._evaluate_weights(optimized_weights, optimization_data)
                
                improvement = optimized_score - current_score
                improvement_percent = (improvement / current_score * 100) if current_score > 0 else 0
                
                # Calcular confianza basada en la mejora y estabilidad
                confidence = self._calculate_optimization_confidence(
                    opt_history, improvement_percent
                )
                
                optimization_results.update({
                    'optimized_weights': optimized_weights,
                    'performance_improvement': improvement_percent,
                    'optimization_history': opt_history,
                    'convergence_achieved': len(opt_history) < max_iterations,
                    'total_iterations': len(opt_history),
                    'confidence_score': confidence
                })
                
                logger.info(f"Weight optimization completed: {improvement_percent:+.2f}% improvement")
            else:
                logger.warning("Weight optimization failed to find better solution")
            
            execution_time = datetime.now() - start_time
            optimization_results['execution_time_seconds'] = execution_time.total_seconds()
            
            return optimization_results
            
        except Exception as e:
            execution_time = datetime.now() - start_time
            optimization_results['execution_time_seconds'] = execution_time.total_seconds()
            optimization_results['error'] = str(e)
            
            logger.error(f"Error during advanced weight optimization: {e}")
            return optimization_results
    
    def _prepare_optimization_data(self, performance_data: Dict[str, Any]) -> Optional[Dict]:
        """Prepara datos para la optimización."""
        try:
            # Obtener datos de validación histórica
            historical_validations = self._get_historical_validations()
            
            if len(historical_validations) < 10:
                logger.warning("Insufficient historical validation data")
                return None
            
            # Preparar conjunto de test scenarios
            test_scenarios = self._generate_test_scenarios()
            
            optimization_data = {
                'historical_validations': historical_validations,
                'test_scenarios': test_scenarios,
                'performance_metrics': performance_data,
                'validation_draws': self.historical_data.tail(50)  # Últimos 50 sorteos para validación
            }
            
            return optimization_data
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            return None
    
    def _get_historical_validations(self) -> List[Dict]:
        """Obtiene datos de validaciones históricas."""
        try:
            # Simular validaciones históricas usando datos reales
            validations = []
            
            recent_draws = self.historical_data.tail(30)
            
            for i, (_, draw) in enumerate(recent_draws.iterrows()):
                # Simular diferentes configuraciones de pesos
                for weight_config in self._generate_weight_configurations():
                    # Evaluar qué tan bien esta configuración habría predicho este sorteo
                    actual_numbers = [draw[f'n{j}'] for j in range(1, 6)]
                    actual_pb = draw['pb']
                    
                    # Simular probabilidades (en implementación real usarías el modelo)
                    simulated_score = self._simulate_prediction_score(
                        actual_numbers, actual_pb, weight_config
                    )
                    
                    validation = {
                        'draw_index': i,
                        'weight_config': weight_config,
                        'prediction_score': simulated_score,
                        'actual_numbers': actual_numbers,
                        'actual_pb': actual_pb
                    }
                    validations.append(validation)
            
            return validations
            
        except Exception as e:
            logger.error(f"Error getting historical validations: {e}")
            return []
    
    def _generate_weight_configurations(self) -> List[Dict]:
        """Genera configuraciones de pesos para testing."""
        configurations = []
        
        # Configuración base
        configurations.append({
            'probability': 0.40,
            'diversity': 0.25,
            'historical': 0.20,
            'risk_adjusted': 0.15
        })
        
        # Variaciones sistemáticas
        variations = [
            {'probability': 0.50, 'diversity': 0.20, 'historical': 0.20, 'risk_adjusted': 0.10},
            {'probability': 0.35, 'diversity': 0.30, 'historical': 0.25, 'risk_adjusted': 0.10},
            {'probability': 0.45, 'diversity': 0.15, 'historical': 0.25, 'risk_adjusted': 0.15},
            {'probability': 0.30, 'diversity': 0.35, 'historical': 0.20, 'risk_adjusted': 0.15},
        ]
        
        configurations.extend(variations)
        return configurations
    
    def _generate_test_scenarios(self) -> List[Dict]:
        """Genera escenarios de test para optimización."""
        scenarios = []
        
        # Escenarios basados en sorteos históricos reales
        recent_draws = self.historical_data.tail(20)
        
        for _, draw in recent_draws.iterrows():
            scenario = {
                'actual_numbers': [draw[f'n{j}'] for j in range(1, 6)],
                'actual_pb': draw['pb'],
                'draw_date': draw['draw_date']
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _simulate_prediction_score(self, actual_numbers: List[int], 
                                 actual_pb: int, weight_config: Dict) -> float:
        """Simula el score de predicción para una configuración de pesos."""
        try:
            # Simular probabilidades del modelo (en implementación real usarías el modelo real)
            wb_probs = {i: np.random.beta(2, 10) for i in range(1, 70)}  # Distribución sesgada
            pb_probs = {i: np.random.beta(2, 20) for i in range(1, 27)}
            
            # Dar probabilidades más altas a los números que realmente salieron
            for num in actual_numbers:
                wb_probs[num] *= 2.0  # Boost para números ganadores
            pb_probs[actual_pb] *= 2.0
            
            # Normalizar
            wb_sum = sum(wb_probs.values())
            pb_sum = sum(pb_probs.values())
            wb_probs = {k: v/wb_sum for k, v in wb_probs.items()}
            pb_probs = {k: v/pb_sum for k, v in pb_probs.items()}
            
            # Aplicar configuración de pesos temporalmente
            original_weights = self.scorer.weights.copy()
            self.scorer.weights = weight_config
            
            # Calcular score
            scores = self.scorer.calculate_total_score(actual_numbers, actual_pb, wb_probs, pb_probs)
            
            # Restaurar pesos originales
            self.scorer.weights = original_weights
            
            return scores['total']
            
        except Exception as e:
            logger.error(f"Error simulating prediction score: {e}")
            return 0.0
    
    def _evaluate_weights(self, weights: Dict[str, float], optimization_data: Dict) -> float:
        """Evalúa una configuración de pesos."""
        try:
            total_score = 0.0
            scenario_count = 0
            
            for scenario in optimization_data['test_scenarios']:
                # Simular score con estos pesos
                score = self._simulate_prediction_score(
                    scenario['actual_numbers'],
                    scenario['actual_pb'],
                    weights
                )
                total_score += score
                scenario_count += 1
            
            return total_score / scenario_count if scenario_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error evaluating weights: {e}")
            return 0.0
    
    def _optimize_differential_evolution(self, current_weights: Dict, 
                                       optimization_data: Dict, 
                                       max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización usando Differential Evolution."""
        logger.info("Running Differential Evolution optimization...")
        
        history = []
        
        def objective(x):
            weights = {
                'probability': x[0],
                'diversity': x[1], 
                'historical': x[2],
                'risk_adjusted': x[3]
            }
            
            # Normalizar para que sumen 1
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            score = self._evaluate_weights(weights, optimization_data)
            history.append({'weights': weights.copy(), 'score': score})
            
            return -score  # Minimizar (DE minimiza)
        
        # Definir bounds
        bounds = [
            self.weight_bounds['probability'],
            self.weight_bounds['diversity'],
            self.weight_bounds['historical'],
            self.weight_bounds['risk_adjusted']
        ]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iterations,
                popsize=15,
                seed=42
            )
            
            if result.success:
                optimized_weights = {
                    'probability': result.x[0],
                    'diversity': result.x[1],
                    'historical': result.x[2],
                    'risk_adjusted': result.x[3]
                }
                return optimized_weights, history
            else:
                logger.warning("Differential Evolution did not converge")
                return None, history
                
        except Exception as e:
            logger.error(f"Error in Differential Evolution: {e}")
            return None, history
    
    def _optimize_bayesian(self, current_weights: Dict, 
                          optimization_data: Dict, 
                          max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización Bayesiana usando Gaussian Process."""
        logger.info("Running Bayesian Optimization...")
        
        history = []
        
        # Preparar datos iniciales
        X_samples = []
        y_samples = []
        
        # Generar muestras iniciales
        for _ in range(10):
            sample_weights = self._generate_random_weights()
            score = self._evaluate_weights(sample_weights, optimization_data)
            
            X_samples.append([
                sample_weights['probability'],
                sample_weights['diversity'],
                sample_weights['historical'],
                sample_weights['risk_adjusted']
            ])
            y_samples.append(score)
            history.append({'weights': sample_weights.copy(), 'score': score})
        
        # Configurar Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        
        best_weights = None
        best_score = max(y_samples)
        
        for iteration in range(max_iterations - 10):
            # Entrenar GP
            gp.fit(X_samples, y_samples)
            
            # Acquisition function (Upper Confidence Bound)
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                return -(mu + 2.0 * sigma)  # UCB con kappa=2.0
            
            # Optimizar acquisition function
            bounds = [
                self.weight_bounds['probability'],
                self.weight_bounds['diversity'],
                self.weight_bounds['historical'],
                self.weight_bounds['risk_adjusted']
            ]
            
            try:
                result = minimize(
                    acquisition,
                    x0=np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    next_x = result.x
                    next_weights = {
                        'probability': next_x[0],
                        'diversity': next_x[1],
                        'historical': next_x[2],
                        'risk_adjusted': next_x[3]
                    }
                    next_weights = self._validate_and_normalize_weights(next_weights)
                    
                    next_score = self._evaluate_weights(next_weights, optimization_data)
                    
                    # Actualizar datos
                    X_samples = np.vstack([X_samples, next_x])
                    y_samples = np.append(y_samples, next_score)
                    
                    history.append({'weights': next_weights.copy(), 'score': next_score})
                    
                    if next_score > best_score:
                        best_score = next_score
                        best_weights = next_weights.copy()
                
            except Exception as e:
                logger.warning(f"Error in Bayesian iteration {iteration}: {e}")
                continue
        
        return best_weights, history
    
    def _optimize_random_search(self, current_weights: Dict, 
                               optimization_data: Dict, 
                               max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización por búsqueda aleatoria."""
        logger.info("Running Random Search optimization...")
        
        history = []
        best_weights = None
        best_score = -float('inf')
        
        for iteration in range(max_iterations):
            # Generar pesos aleatorios
            random_weights = self._generate_random_weights()
            score = self._evaluate_weights(random_weights, optimization_data)
            
            history.append({'weights': random_weights.copy(), 'score': score})
            
            if score > best_score:
                best_score = score
                best_weights = random_weights.copy()
        
        return best_weights, history
    
    def _optimize_gradient_descent(self, current_weights: Dict, 
                                  optimization_data: Dict, 
                                  max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización por descenso de gradiente."""
        logger.info("Running Gradient Descent optimization...")
        
        history = []
        
        def objective(x):
            weights = {
                'probability': x[0],
                'diversity': x[1],
                'historical': x[2],
                'risk_adjusted': x[3]
            }
            weights = self._validate_and_normalize_weights(weights)
            score = self._evaluate_weights(weights, optimization_data)
            history.append({'weights': weights.copy(), 'score': score})
            return -score  # Minimizar
        
        # Punto inicial
        x0 = [
            current_weights['probability'],
            current_weights['diversity'], 
            current_weights['historical'],
            current_weights['risk_adjusted']
        ]
        
        bounds = [
            self.weight_bounds['probability'],
            self.weight_bounds['diversity'],
            self.weight_bounds['historical'],
            self.weight_bounds['risk_adjusted']
        ]
        
        try:
            result = minimize(
                objective,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations}
            )
            
            if result.success:
                optimized_weights = {
                    'probability': result.x[0],
                    'diversity': result.x[1],
                    'historical': result.x[2],
                    'risk_adjusted': result.x[3]
                }
                return optimized_weights, history
            else:
                return None, history
                
        except Exception as e:
            logger.error(f"Error in Gradient Descent: {e}")
            return None, history
    
    def _optimize_genetic_algorithm(self, current_weights: Dict, 
                                   optimization_data: Dict, 
                                   max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización usando Algoritmo Genético simple."""
        logger.info("Running Genetic Algorithm optimization...")
        
        history = []
        population_size = 20
        mutation_rate = 0.1
        
        # Inicializar población
        population = []
        for _ in range(population_size):
            individual = self._generate_random_weights()
            score = self._evaluate_weights(individual, optimization_data)
            population.append({'weights': individual, 'score': score, 'fitness': score})
            history.append({'weights': individual.copy(), 'score': score})
        
        for generation in range(max_iterations // population_size):
            # Selección (torneo)
            parents = []
            for _ in range(population_size // 2):
                tournament = np.random.choice(population, size=3, replace=False)
                winner = max(tournament, key=lambda x: x['fitness'])
                parents.append(winner['weights'])
            
            # Crossover y mutación
            new_population = []
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutación
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                
                # Evaluar hijos
                for child in [child1, child2]:
                    score = self._evaluate_weights(child, optimization_data)
                    new_population.append({'weights': child, 'score': score, 'fitness': score})
                    history.append({'weights': child.copy(), 'score': score})
            
            # Reemplazar población
            population = sorted(new_population, key=lambda x: x['fitness'], reverse=True)[:population_size]
        
        # Retornar el mejor individuo
        best_individual = max(population, key=lambda x: x['fitness'])
        return best_individual['weights'], history
    
    def _optimize_ensemble(self, current_weights: Dict, 
                          optimization_data: Dict, 
                          max_iterations: int) -> Tuple[Optional[Dict], List]:
        """Optimización ensemble que combina múltiples algoritmos."""
        logger.info("Running Ensemble optimization...")
        
        algorithms = ['differential_evolution', 'bayesian_optimization', 'random_search']
        all_results = []
        all_history = []
        
        iterations_per_algo = max_iterations // len(algorithms)
        
        for algo in algorithms:
            logger.info(f"Running {algo} as part of ensemble...")
            try:
                optimizer_func = self.algorithms[algo]
                weights, history = optimizer_func(current_weights, optimization_data, iterations_per_algo)
                
                if weights is not None:
                    score = self._evaluate_weights(weights, optimization_data)
                    all_results.append({'algorithm': algo, 'weights': weights, 'score': score})
                
                all_history.extend(history)
                
            except Exception as e:
                logger.error(f"Error in ensemble algorithm {algo}: {e}")
                continue
        
        if all_results:
            # Seleccionar el mejor resultado
            best_result = max(all_results, key=lambda x: x['score'])
            logger.info(f"Best ensemble result from {best_result['algorithm']}: {best_result['score']:.4f}")
            return best_result['weights'], all_history
        
        return None, all_history
    
    def _generate_random_weights(self) -> Dict[str, float]:
        """Genera pesos aleatorios válidos."""
        weights = {}
        for component, (min_val, max_val) in self.weight_bounds.items():
            weights[component] = np.random.uniform(min_val, max_val)
        
        return self._validate_and_normalize_weights(weights)
    
    def _validate_and_normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Valida y normaliza pesos para que sumen 1."""
        # Asegurar que están en bounds
        for component, (min_val, max_val) in self.weight_bounds.items():
            if component in weights:
                weights[component] = np.clip(weights[component], min_val, max_val)
        
        # Normalizar para que sumen 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Operador de crossover para algoritmo genético."""
        alpha = np.random.random()
        
        child1 = {}
        child2 = {}
        
        for component in parent1.keys():
            child1[component] = alpha * parent1[component] + (1 - alpha) * parent2[component]
            child2[component] = (1 - alpha) * parent1[component] + alpha * parent2[component]
        
        return self._validate_and_normalize_weights(child1), self._validate_and_normalize_weights(child2)
    
    def _mutate(self, individual: Dict) -> Dict:
        """Operador de mutación para algoritmo genético."""
        mutated = individual.copy()
        
        # Mutar un componente aleatorio
        component = np.random.choice(list(individual.keys()))
        min_val, max_val = self.weight_bounds[component]
        
        # Mutación gaussiana
        mutation = np.random.normal(0, 0.05)
        mutated[component] += mutation
        
        return self._validate_and_normalize_weights(mutated)
    
    def _calculate_optimization_confidence(self, history: List, improvement_percent: float) -> float:
        """Calcula confianza en la optimización basada en convergencia y mejora."""
        try:
            if len(history) < 5:
                return 0.0
            
            # Factor de mejora
            improvement_factor = min(1.0, improvement_percent / 10.0)  # 10% mejora = confianza 1.0
            
            # Factor de convergencia (estabilidad en últimas iteraciones)
            recent_scores = [h['score'] for h in history[-10:]]
            if len(recent_scores) > 1:
                stability = 1.0 - (np.std(recent_scores) / np.mean(recent_scores))
                stability = max(0.0, stability)
            else:
                stability = 0.0
            
            # Factor de exploración (diversidad de soluciones probadas)
            all_scores = [h['score'] for h in history]
            exploration = min(1.0, np.std(all_scores) / np.mean(all_scores))
            
            # Confianza combinada
            confidence = 0.5 * improvement_factor + 0.3 * stability + 0.2 * exploration
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating optimization confidence: {e}")
            return 0.0
