from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    return_sources: bool = False
    return_images: bool = True

class ImageResponse(BaseModel):
    data: str
    content_type: str = "image/jpeg"

class TableData(BaseModel):
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None

class YSeries(BaseModel):
    name: Optional[str] = None
    values: Optional[List[Union[int, float, str]]] = None

class VisualizationItem(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None  # Expected values: bar, line, pie, etc.
    x_axis: Optional[List[str]] = None
    y_axis: Optional[List[Union[int, float, str]]] = None
    data_labels: Optional[List[str]] = None
    y_series: Optional[List[YSeries]] = None
    table_data: Optional[TableData] = None
    description: Optional[str] = None

class Visualization(BaseModel):
    visualizations: Optional[List[VisualizationItem]] = None

class Comparison(BaseModel):
    compared_values: Optional[List[str]] = None
    basis: Optional[str] = None
    result: Optional[str] = None
    graph_type: Optional[str] = None

class RawTable(BaseModel):
    columns: Optional[List[str]] = None
    data: Optional[List[List[str]]] = None

class TableAnalysis(BaseModel):
    structure: Optional[str] = None
    headers: Optional[List[str]] = None
    row_count: Optional[Union[int, str]] = None
    key_metrics: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    raw_table: Optional[RawTable] = None

class Details(BaseModel):
    key_points: Optional[List[str]] = None
    source_references: Optional[List[str]] = None

class RAGTextResponse(BaseModel):
    answer: str
    details: Details
    table_analysis: TableAnalysis
    comparison: Comparison
    visualization: Visualization

class RAGResponse(BaseModel):
    text_response: RAGTextResponse
    images: Optional[List[ImageResponse]] = None
    sources: Optional[Dict[str, Any]] = None