from pydantic import BaseModel

class ClinicalEvidence(BaseModel):
    citation_id: int
    chunk_id: int
    page: int
    section: str
    text: str
    is_dense: bool = True
    is_sparse: bool = False
