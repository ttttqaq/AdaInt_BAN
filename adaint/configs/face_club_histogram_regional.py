_base_ = './face_club_semantic_generalization.yaml'

exp_name = 'club_histogram_regional_s2026'
work_dir = './work_dirs/club_histogram_regional_s2026'
model = dict(luminance_histogram=dict(mode='regional', weight=0.02))
