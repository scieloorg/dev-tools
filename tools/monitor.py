#!/usr/bin/env python3
"""
Auto Profiler - Análise automática de performance e memória
Dependências mínimas: apenas bibliotecas padrão do Python
"""

import cProfile
import pstats
import tracemalloc
import time
import functools
import sys
import io
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import linecache
import os

class AutoProfiler:
    """Analisador automático de performance e memória"""
    
    def __init__(self, 
                 top_functions=10,
                 memory_threshold_mb=1.0,
                 time_threshold_seconds=0.1):
        """
        Args:
            top_functions: Número de funções a mostrar
            memory_threshold_mb: Limiar para reportar uso de memória
            time_threshold_seconds: Limiar para reportar tempo de execução
        """
        self.top_functions = top_functions
        self.memory_threshold_mb = memory_threshold_mb
        self.time_threshold_seconds = time_threshold_seconds
        self.profile_data = {}
        self.memory_snapshots = []
        
    def profile(self, func=None, label=None):
        """Decorator/context manager para profiling automático"""
        if func is None:
            return self._profile_context(label)
        return self._profile_decorator(func)
    
    def _profile_decorator(self, func):
        """Decorator para funções"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label = f"{func.__module__}.{func.__name__}"
            
            # Inicia profiling de memória
            tracemalloc.start()
            mem_before = tracemalloc.get_traced_memory()[0]
            
            # Inicia profiling de tempo
            profiler = cProfile.Profile()
            profiler.enable()
            start_time = time.perf_counter()
            
            try:
                # Executa função
                result = func(*args, **kwargs)
                
                # Para profiling
                end_time = time.perf_counter()
                profiler.disable()
                mem_after = tracemalloc.get_traced_memory()[0]
                
                # Salva dados
                self._save_profile_data(
                    label, 
                    profiler, 
                    end_time - start_time,
                    (mem_after - mem_before) / 1024 / 1024  # MB
                )
                
                return result
                
            finally:
                tracemalloc.stop()
                
        return wrapper
    
    @contextmanager
    def _profile_context(self, label):
        """Context manager para blocos de código"""
        if label is None:
            label = "code_block"
            
        # Inicia profiling
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Para profiling
            end_time = time.perf_counter()
            profiler.disable()
            mem_after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            
            # Salva dados
            self._save_profile_data(
                label,
                profiler,
                end_time - start_time,
                (mem_after - mem_before) / 1024 / 1024
            )
    
    def _save_profile_data(self, label, profiler, total_time, memory_mb):
        """Salva dados do profiling"""
        self.profile_data[label] = {
            'profiler': profiler,
            'total_time': total_time,
            'memory_mb': memory_mb
        }
    
    def analyze_and_report(self):
        """Analisa todos os dados e gera relatório"""
        if not self.profile_data:
            print("Nenhum dado de profiling encontrado!")
            return
            
        print("\n" + "="*70)
        print("RELATÓRIO DE ANÁLISE AUTOMÁTICA")
        print("="*70)
        
        # Resumo geral
        self._print_summary()
        
        # Problemas identificados
        problems = self._identify_problems()
        if problems:
            self._print_problems(problems)
        
        # Detalhes por função
        self._print_detailed_analysis()
        
        # Recomendações
        self._print_recommendations(problems)
    
    def _print_summary(self):
        """Imprime resumo geral"""
        print("\n📊 RESUMO GERAL:")
        print("-" * 50)
        
        total_time = sum(d['total_time'] for d in self.profile_data.values())
        total_memory = sum(d['memory_mb'] for d in self.profile_data.values())
        
        print(f"Tempo total: {total_time:.3f}s")
        print(f"Memória total: {total_memory:.2f}MB")
        print(f"Funções analisadas: {len(self.profile_data)}")
    
    def _identify_problems(self):
        """Identifica problemas de performance e memória"""
        problems = {
            'slow_functions': [],
            'memory_hungry': [],
            'frequent_calls': [],
            'inefficient': []
        }
        
        for label, data in self.profile_data.items():
            # Funções lentas
            if data['total_time'] > self.time_threshold_seconds:
                problems['slow_functions'].append({
                    'function': label,
                    'time': data['total_time']
                })
            
            # Alto uso de memória
            if data['memory_mb'] > self.memory_threshold_mb:
                problems['memory_hungry'].append({
                    'function': label,
                    'memory': data['memory_mb']
                })
            
            # Analisa estatísticas detalhadas
            stats = pstats.Stats(data['profiler'])
            stats.sort_stats('cumulative')
            
            # Identifica chamadas frequentes
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                if nc > 10000:  # Mais de 10k chamadas
                    problems['frequent_calls'].append({
                        'function': self._format_func_name(func_info),
                        'calls': nc,
                        'total_time': ct
                    })
                
                # Identifica funções ineficientes (muito tempo por chamada)
                if nc > 0 and (ct / nc) > 0.001:  # Mais de 1ms por chamada
                    problems['inefficient'].append({
                        'function': self._format_func_name(func_info),
                        'time_per_call': ct / nc,
                        'total_calls': nc
                    })
        
        return problems
    
    def _print_problems(self, problems):
        """Imprime problemas identificados"""
        print("\n🚨 PROBLEMAS IDENTIFICADOS:")
        print("-" * 50)
        
        # Funções lentas
        if problems['slow_functions']:
            print("\n⏱️  Funções Lentas (>{:.2f}s):".format(self.time_threshold_seconds))
            for p in sorted(problems['slow_functions'], key=lambda x: x['time'], reverse=True)[:5]:
                print(f"   • {p['function']}: {p['time']:.3f}s")
        
        # Alto uso de memória
        if problems['memory_hungry']:
            print("\n💾 Alto Uso de Memória (>{:.1f}MB):".format(self.memory_threshold_mb))
            for p in sorted(problems['memory_hungry'], key=lambda x: x['memory'], reverse=True)[:5]:
                print(f"   • {p['function']}: {p['memory']:.2f}MB")
        
        # Chamadas frequentes
        if problems['frequent_calls']:
            print("\n🔄 Funções com Muitas Chamadas (>10k):")
            unique_funcs = {}
            for p in problems['frequent_calls']:
                key = p['function']
                if key not in unique_funcs or p['calls'] > unique_funcs[key]['calls']:
                    unique_funcs[key] = p
            
            for p in sorted(unique_funcs.values(), key=lambda x: x['calls'], reverse=True)[:5]:
                print(f"   • {p['function']}: {p['calls']:,} chamadas ({p['total_time']:.3f}s total)")
        
        # Funções ineficientes
        if problems['inefficient']:
            print("\n⚠️  Funções Ineficientes (>1ms/chamada):")
            unique_funcs = {}
            for p in problems['inefficient']:
                key = p['function']
                if key not in unique_funcs or p['time_per_call'] > unique_funcs[key]['time_per_call']:
                    unique_funcs[key] = p
            
            for p in sorted(unique_funcs.values(), key=lambda x: x['time_per_call'], reverse=True)[:5]:
                print(f"   • {p['function']}: {p['time_per_call']*1000:.2f}ms/chamada ({p['total_calls']} chamadas)")
    
    def _print_detailed_analysis(self):
        """Imprime análise detalhada por função"""
        print("\n📈 ANÁLISE DETALHADA:")
        print("-" * 50)
        
        for label, data in sorted(self.profile_data.items(), 
                                 key=lambda x: x[1]['total_time'], 
                                 reverse=True):
            print(f"\n▶ {label}")
            print(f"  Tempo: {data['total_time']:.3f}s | Memória: {data['memory_mb']:.2f}MB")
            
            # Top funções internas
            stats = pstats.Stats(data['profiler'])
            stats.sort_stats('cumulative')
            
            # Captura saída
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            stats.print_stats(5)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Processa e exibe apenas linhas relevantes
            lines = output.split('\n')
            for line in lines[5:10]:  # Pula cabeçalho
                if line.strip():
                    # Simplifica saída
                    parts = line.split()
                    if len(parts) >= 6:
                        calls = parts[0]
                        time = parts[2]
                        func = ' '.join(parts[5:])
                        if float(time) > 0.001:  # Só mostra se > 1ms
                            print(f"    - {func}: {time}s ({calls} calls)")
    
    def _print_recommendations(self, problems):
        """Gera recomendações baseadas nos problemas"""
        print("\n💡 RECOMENDAÇÕES:")
        print("-" * 50)
        
        recommendations = []
        
        # Recomendações para funções lentas
        if problems['slow_functions']:
            recommendations.append(
                "• Otimize as funções lentas usando algoritmos mais eficientes ou cache"
            )
        
        # Recomendações para memória
        if problems['memory_hungry']:
            recommendations.append(
                "• Considere processar dados em chunks ou usar geradores para reduzir memória"
            )
        
        # Recomendações para chamadas frequentes
        if problems['frequent_calls']:
            recommendations.append(
                "• Use memoização ou cache para funções chamadas frequentemente"
            )
            recommendations.append(
                "• Considere vetorização com NumPy para operações repetitivas"
            )
        
        # Recomendações para funções ineficientes
        if problems['inefficient']:
            recommendations.append(
                "• Revise funções com alto tempo por chamada - possível complexidade O(n²) ou pior"
            )
        
        if recommendations:
            for rec in recommendations:
                print(rec)
        else:
            print("✅ Nenhum problema significativo encontrado!")
    
    def _format_func_name(self, func_info):
        """Formata nome da função de forma legível"""
        filename, line, func_name = func_info
        if filename.startswith('<'):
            return func_name
        return f"{Path(filename).name}:{line}({func_name})"
    
    def save_report(self, filename="profiling_report.txt"):
        """Salva relatório em arquivo"""
        old_stdout = sys.stdout
        try:
            with open(filename, 'w') as f:
                sys.stdout = f
                self.analyze_and_report()
        finally:
            sys.stdout = old_stdout
        
        # Print após restaurar stdout
        print(f"\n📄 Relatório salvo em: {filename}")


# Função auxiliar para profile rápido
def quick_profile(func):
    """Decorator para profiling rápido de uma função"""
    profiler = AutoProfiler()
    wrapped = profiler.profile(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = wrapped(*args, **kwargs)
        profiler.analyze_and_report()
        return result
    
    return wrapper


# Exemplo de uso
if __name__ == "__main__":
    import random
    
    # Cria profiler
    profiler = AutoProfiler(
        top_functions=5,
        memory_threshold_mb=0.5,
        time_threshold_seconds=0.05
    )
    
    # Exemplo 1: Função com problema de performance
    @profiler.profile
    def processar_lista_ineficiente(n=10000):
        """Exemplo de código ineficiente"""
        resultado = []
        for i in range(n):
            # Operação O(n²) - append em lista é O(1) mas o loop aninhado não
            temp = []
            for j in range(100):
                temp.append(i * j)
            resultado.append(sum(temp))
        return resultado
    
    # Exemplo 2: Função com uso excessivo de memória
    @profiler.profile
    def criar_matriz_grande(size=1000):
        """Exemplo de alto uso de memória"""
        matriz = [[random.random() for _ in range(size)] for _ in range(size)]
        return sum(sum(linha) for linha in matriz)
    
    # Exemplo 3: Usando context manager
    with profiler.profile(label="processamento_complexo"):
        # Simula processamento
        dados = [i**2 for i in range(50000)]
        resultado = sum(dados) / len(dados)
    
    # Executa funções
    print("Executando análise...")
    processar_lista_ineficiente(5000)
    criar_matriz_grande(500)
    
    # Gera relatório
    profiler.analyze_and_report()
    
    # Salva em arquivo
    profiler.save_report()