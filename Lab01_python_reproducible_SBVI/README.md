# Laboratorio 1: Python científico reproducible y legible

## 1. ¿Qué problema resuelve el proyecto?
Este proyecto implementa un pipeline de análisis de señales sintéticas completamente reproducible y automatizado. Su objetivo es garantizar que la ejecución de análisis numéricos, validaciones estadísticas, control de calidad y generación de reportes no dependa de configuraciones manuales ni de un equipo específico, cumpliendo con los estándares de ingeniería de software reproducible.

## 2. ¿Qué versión de Python solicita?
- Python: >= 3.13

## 3. ¿Qué comando reconstruye el entorno?
Para sincronizar las dependencias de forma exacta mediante el gestor de entornos, ejecute: uv sync --locked
Para verificar errores de tipeo, en el código, o simplemente de formato (que siga reglas de Python): uv run ruff check . 
y uv run ruff format --check .

## 4. ¿Qué comando prueba?
Para ejecutar la suite completa de pruebas unitarias con pytest, ejecute: uv run pytest

## 5. ¿Qué comando genera resultados?
Para ejecutar el script principal que procesa las señales, valida canales, realiza operaciones de álgebra lineal y genera todos los artefactos (.json, .tex, .png), ejecute: uv run python scripts/reproduce.py

## 6. ¿Qué comandos compilan el informe y la presentación?
Para compilar los documentos finales en formato PDF utilizando latexmk, ejecute:
- Informe técnico: latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/reports reports/informe.tex
    - Abrir proceso desde la consola para...
        - Windows: Start-Process "build\reports\informe.pdf"
        - Linux: xdg-open build/reports/informe.pdf
- Presentación (Beamer): latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/slides slides/presentacion.tex
    - Abrir proceso desde la consola para...
        - Windows: Start-Process "build\slides\presentacion.pdf"
        - Linux: xdg-open build/slides/presentacion.pdf

--Para interrumpir el proceso en caso de error: latexmk -c slides/presentacion.tex

## 7. ¿Dónde quedan datos, figuras y reportes?
- Datos originales / sintéticos: Generados en el flujo del pipeline.
- Figuras generadas: figures/
- Tablas y artefactos numéricos: results/
- PDFs finales compilados: build/reports/ y build/slides/

## 8. ¿Qué supuestos y limitaciones existen?
- Supuestos: El entorno de ejecución cuenta con Python y uv instalados. Se utiliza una semilla determinista (seed = 42) para garantizar la reproducibilidad numérica.
- Limitaciones: El sistema opera estrictamente sobre señales sintéticas controladas. Por tanto, no es extrapolable a datos fisiológicos reales, ni a entornos clínicos. Así mismo, la manipulación inicial de herramientas automatizadas presentó una curva de aprendizaje operativa para el equipo.