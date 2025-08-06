
#!/usr/bin/env python3
"""
SHIOL+ Cleanup Script - Eliminación de Archivos No Utilizados
============================================================

Script para identificar y eliminar archivos huérfanos que no se utilizan
en el sistema actual SHIOL+ v5.1
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Set
import ast
import re

class ProjectCleaner:
    def __init__(self):
        self.project_root = Path(".")
        self.used_files = set()
        self.unused_files = []
        self.safe_to_delete = []
        
        # Archivos core que nunca se deben eliminar
        self.protected_files = {
            "main.py",
            "requirements.txt",
            ".replit",
            ".gitignore",
            "README.md",
            "config/config.ini"
        }
        
        # Directorios que contienen archivos generados/temporales
        self.temp_directories = {
            "logs/",
            "data/validations/",
            "reports/",
            "__pycache__/",
            ".pytest_cache/"
        }
    
    def analyze_imports(self, file_path: Path) -> Set[str]:
        """Analiza las importaciones en un archivo Python."""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Buscar imports relativos del proyecto
            import_patterns = [
                r'from src\.(\w+)',  # from src.module
                r'import src\.(\w+)', # import src.module
                r'from \.(\w+)',     # from .module
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    imports.add(f"src/{match}.py")
                    
        except Exception as e:
            print(f"Error analizando {file_path}: {e}")
            
        return imports
    
    def find_used_files(self):
        """Encuentra todos los archivos que están siendo utilizados."""
        print("🔍 Analizando archivos utilizados en el proyecto...")
        
        # Empezar desde main.py y archivos de entrada
        entry_points = [
            "main.py",
            "src/api.py",
            "src/public_api.py",
            "src/cli.py"
        ]
        
        to_analyze = entry_points.copy()
        
        while to_analyze:
            current_file = to_analyze.pop(0)
            if current_file in self.used_files:
                continue
                
            file_path = Path(current_file)
            if not file_path.exists():
                continue
                
            self.used_files.add(current_file)
            
            # Analizar importaciones
            imports = self.analyze_imports(file_path)
            for imported_file in imports:
                if imported_file not in self.used_files:
                    to_analyze.append(imported_file)
    
    def identify_unused_files(self):
        """Identifica archivos no utilizados."""
        print("📋 Identificando archivos no utilizados...")
        
        # Buscar todos los archivos Python en src/
        src_files = list(Path("src").glob("*.py"))
        
        for src_file in src_files:
            relative_path = str(src_file)
            if relative_path not in self.used_files and relative_path not in self.protected_files:
                self.unused_files.append(relative_path)
        
        # Archivos específicos que pueden ser innecesarios
        potentially_unused = [
            "src/notifications.py",  # No se usa en el pipeline actual
            "src/performance_evaluator.py",  # Duplica funcionalidad
            "src/auto_retrainer.py",  # No se integra actualmente
            "src/enhanced_weight_optimizer.py",  # No se usa
            "src/model_validator.py",  # Funcionalidad duplicada
            "src/pipeline_logger.py",  # No se integra
            "src/scheduler.py",  # Se maneja desde main.py
        ]
        
        for file_path in potentially_unused:
            if Path(file_path).exists() and file_path not in self.used_files:
                self.safe_to_delete.append(file_path)
    
    def clean_temp_directories(self):
        """Limpia directorios temporales."""
        print("🧹 Limpiando directorios temporales...")
        
        for temp_dir in self.temp_directories:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                if temp_dir.endswith("/"):
                    # Limpiar contenido del directorio
                    for item in temp_path.iterdir():
                        if item.is_file():
                            try:
                                item.unlink()
                                print(f"   ✓ Eliminado: {item}")
                            except Exception as e:
                                print(f"   ✗ Error eliminando {item}: {e}")
    
    def clean_old_reports(self):
        """Elimina reportes antiguos, manteniendo solo los 5 más recientes."""
        print("📊 Limpiando reportes antiguos...")
        
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return
        
        # Obtener todos los archivos de reporte
        report_files = list(reports_dir.glob("pipeline_report_*.json"))
        
        if len(report_files) <= 5:
            print("   ℹ️ Solo hay 5 o menos reportes, no se elimina ninguno")
            return
        
        # Ordenar por fecha de modificación (más recientes primero)
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Eliminar todos excepto los 5 más recientes
        for old_report in report_files[5:]:
            try:
                old_report.unlink()
                print(f"   ✓ Eliminado reporte antiguo: {old_report.name}")
            except Exception as e:
                print(f"   ✗ Error eliminando {old_report}: {e}")
    
    def generate_cleanup_report(self) -> str:
        """Genera un reporte de la limpieza."""
        report_lines = [
            "# SHIOL+ Cleanup Report - Archivos No Utilizados",
            f"## Fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Archivos Identificados como No Utilizados:",
            ""
        ]
        
        if self.safe_to_delete:
            for file_path in self.safe_to_delete:
                file_size = 0
                try:
                    file_size = Path(file_path).stat().st_size
                except:
                    pass
                
                report_lines.append(f"- `{file_path}` ({file_size} bytes)")
                
                # Añadir razón por la que no se usa
                reasons = {
                    "src/notifications.py": "Sistema de notificaciones no integrado al pipeline actual",
                    "src/performance_evaluator.py": "Funcionalidad duplicada con evaluator.py",
                    "src/auto_retrainer.py": "Sistema de reentrenamiento automático no activo",
                    "src/enhanced_weight_optimizer.py": "Optimizador avanzado no utilizado",
                    "src/model_validator.py": "Validación duplicada con otros componentes",
                    "src/pipeline_logger.py": "Logger especializado no integrado",
                    "src/scheduler.py": "Funcionalidad de scheduler manejada desde main.py"
                }
                
                if file_path in reasons:
                    report_lines.append(f"  - **Razón**: {reasons[file_path]}")
                    
        else:
            report_lines.append("- No se encontraron archivos seguros para eliminar")
        
        report_lines.extend([
            "",
            "## Recomendaciones:",
            "",
            "1. **Revisar manualmente** cada archivo antes de eliminarlo",
            "2. **Crear backup** del proyecto antes de eliminar archivos",
            "3. **Verificar** que no hay dependencias ocultas",
            "4. **Probar** el sistema después de la limpieza",
            "",
            "## Archivos Protegidos (NO eliminar):",
            ""
        ])
        
        for protected in self.protected_files:
            report_lines.append(f"- `{protected}`")
        
        return "\n".join(report_lines)
    
    def run_analysis(self):
        """Ejecuta el análisis completo."""
        print("🚀 Iniciando análisis de limpieza de SHIOL+...")
        print("=" * 50)
        
        # Análisis de archivos utilizados
        self.find_used_files()
        print(f"✓ Encontrados {len(self.used_files)} archivos en uso")
        
        # Identificar archivos no utilizados
        self.identify_unused_files()
        print(f"⚠️ Identificados {len(self.safe_to_delete)} archivos potencialmente no utilizados")
        
        # Mostrar archivos seguros para eliminar
        if self.safe_to_delete:
            print("\n📋 Archivos seguros para eliminar:")
            for file_path in self.safe_to_delete:
                try:
                    size = Path(file_path).stat().st_size
                    print(f"   - {file_path} ({size} bytes)")
                except:
                    print(f"   - {file_path} (tamaño desconocido)")
        
        # Generar reporte
        report = self.generate_cleanup_report()
        report_path = "cleanup_unused_files_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📝 Reporte generado: {report_path}")
        
        return self.safe_to_delete
    
    def execute_cleanup(self, confirm: bool = False):
        """Ejecuta la limpieza de archivos."""
        if not confirm:
            print("\n⚠️ ADVERTENCIA: Esta operación eliminará archivos permanentemente")
            response = input("¿Deseas continuar? (y/N): ").lower()
            if response != 'y':
                print("❌ Operación cancelada")
                return False
        
        print("\n🧹 Ejecutando limpieza...")
        
        # Limpiar archivos no utilizados
        deleted_count = 0
        for file_path in self.safe_to_delete:
            try:
                Path(file_path).unlink()
                print(f"   ✓ Eliminado: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"   ✗ Error eliminando {file_path}: {e}")
        
        # Limpiar directorios temporales
        self.clean_temp_directories()
        
        # Limpiar reportes antiguos
        self.clean_old_reports()
        
        print(f"\n✅ Limpieza completada. {deleted_count} archivos eliminados.")
        return True

def main():
    """Función principal."""
    cleaner = ProjectCleaner()
    
    # Ejecutar análisis
    files_to_delete = cleaner.run_analysis()
    
    if not files_to_delete:
        print("\n✅ No se encontraron archivos seguros para eliminar.")
        print("   El proyecto ya está limpio.")
        return
    
    print("\n" + "=" * 50)
    print("💡 OPCIONES:")
    print("1. Solo generar reporte (ya completado)")
    print("2. Ejecutar limpieza automática")
    print("3. Salir sin cambios")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == "2":
            cleaner.execute_cleanup()
        elif choice == "3":
            print("❌ Operación cancelada")
        else:
            print("ℹ️ Solo se generó el reporte. Revisa 'cleanup_unused_files_report.md'")
            
    except KeyboardInterrupt:
        print("\n❌ Operación cancelada por el usuario")

if __name__ == "__main__":
    main()
