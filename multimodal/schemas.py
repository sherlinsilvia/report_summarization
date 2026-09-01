"""
Structured Data Schemas for Multimodal Medical Evidence & Clinical Summarization.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class MedicationItem(BaseModel):
    name: str = Field(description="Name of the prescribed medicine")
    strength: Optional[str] = Field(default=None, description="Strength / concentration e.g. 500mg, 10mg")
    dosage: Optional[str] = Field(default=None, description="Dosage quantity e.g. 1 tablet, 5ml, 2 drops")
    frequency: Optional[str] = Field(default=None, description="Frequency e.g. Once daily, Twice daily, Morning and night")
    route: Optional[str] = Field(default="Oral", description="Administration route e.g. Oral, IV, Topical")
    food_instruction: Optional[str] = Field(default=None, description="Before food, After food, With water")
    duration: Optional[str] = Field(default=None, description="Duration e.g. 5 days, 10 days, 1 month")
    special_instructions: Optional[str] = Field(default=None, description="Any specific precautions or notes")
    is_uncertain: bool = Field(default=False, description="True if handwriting or medicine name was difficult to read")
    source: str = Field(default="Prescription", description="Source document or image name")

class ImageFindingItem(BaseModel):
    modality: str = Field(description="X-ray, CT Scan, MRI, Clinical Photograph, Ultrasound")
    anatomical_region: str = Field(description="Chest, Brain, Knee, Abdomen, Spine, etc.")
    observation: str = Field(description="Visual findings observed in the image")
    confidence: str = Field(default="Medium", description="High, Medium, Low, or Inconclusive")
    potential_implications: Optional[str] = Field(default=None, description="Possible clinical considerations")
    limitations: str = Field(
        default="AI-assisted observation only. Not a definitive diagnosis; requires qualified radiological review.",
        description="Clinical safety limitation"
    )
    source: str = Field(default="Medical Image", description="Source image filename")

class PatientDemographics(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None
    date: Optional[str] = None
    attending_physician: Optional[str] = None
    clinic_or_hospital: Optional[str] = None

class StructuredClinicalEvidence(BaseModel):
    patient_information: PatientDemographics = Field(default_factory=PatientDemographics)
    document_type: str = "general_medical"
    detected_types: List[str] = Field(default_factory=list)
    date: Optional[str] = None
    clinical_history: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    medications: List[MedicationItem] = Field(default_factory=list)
    investigations: List[str] = Field(default_factory=list)
    image_findings: List[ImageFindingItem] = Field(default_factory=list)
    diagnoses_mentioned: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    follow_up: List[str] = Field(default_factory=list)
    uncertain_information: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)
    raw_extracted_text: str = ""
