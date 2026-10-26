from document_processor.core.registry import Registry
from document_processor.logical.profile import Profile
from document_processor.logical.profiles.vn_regulation import VN_REGULATION

PROFILES: Registry[Profile] = Registry("profile", {VN_REGULATION.name: VN_REGULATION})