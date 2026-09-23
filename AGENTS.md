# AGENTS.md

## Contexto del proyecto

Este repositorio implementa un modelo RCNN para predecir proteínas antigénicas a partir de embeddings por residuo generados con ESM-2. El objetivo de esta guía es mantener la reproducibilidad, la trazabilidad y la modularidad del flujo de entrenamiento, validación e inferencia.

## Regla principal

- La lógica productiva vive en `src/modules/`.
- `scripts/` solo contiene wrappers de ejecución.
- Los notebooks y experimentos no deben contener lógica de producción.
- La configuración personalizable debe vivir en `config/*.yaml`.
- Los artefactos de entrenamiento y validación se guardan bajo `data/` o `results/`.
- Los modelos, métricas y resultados deben ser reproducibles y trazables.

## Estructura esperada

- `src/modules/rcnn_model/`: definición del modelo RCNN.
- `src/modules/rcnn_training/`: entrenamiento, validación y evaluación.
- `src/modules/rcnn_inference/`: predicción con modelos entrenados.
- `config/`: hiperparámetros, dimensiones y configuración del experimento.
- `scripts/`: puntos de entrada de ejecución, sin lógica de negocio.
- `shared/`: utilidades reutilizables y de apoyo.

## Reglas de diseño

1. Una función debe tener una única responsabilidad.
2. No hardcodear rutas locales ni valores de entrenamiento en el código productivo.
3. Los hiperparámetros deben estar en el archivo de configuración YAML.
4. La lógica que construye un modelo debe separarse de la lógica que lo entrena.
5. La lógica de inferencia debe ser independiente del flujo de entrenamiento.
6. Cada módulo debe exponer un punto de entrada claro y reutilizable.

## Configuración y persistencia

- Los valores configurables deben ir en `config/rcnn_config.yaml`.
- El archivo YAML debe concentrar:
  - dimensiones de entrada/salida,
  - filtros y tamaños de kernel,
  - hidden sizes,
  - número de capas, dropout,
  - parámetros de entrenamiento,
  - pesos de clase,
  - rutas de salida y artefactos.
- Las nuevas arquitecturas deben añadirse como variaciones del mismo esquema de configuración, sin duplicar bloques de lógica.

## Restricciones del proyecto

- La arquitectura base de RCNN ya funciona; no se debe romper el flujo funcional actual.
- La comparación de arquitecturas se hace en una etapa posterior y exclusivamente desde una configuración explícita.
- No se introducen APIs ni servicios web para esta fase de trabajo.
- Las modificaciones deben ser pequeñas, legibles y revisables.

## Calidad mínima

Todo cambio en código productivo debe:
- respetar la modularidad del proyecto,
- evitar duplicidades,
- usar configuración centralizada,
- respetar una estructura limpia de paquetes,
- mantenerse compatible con el flujo actual de entrenamiento.

## Commit y revisión

- Los commits deben ser pequeños y temáticos.
- Un commit debe resolver una única intención: modularización, configuración o ajuste de un módulo.
- Las ramas deben seguir un flujo de trabajo claro y las tareas deben revisarse antes de integrarse.
