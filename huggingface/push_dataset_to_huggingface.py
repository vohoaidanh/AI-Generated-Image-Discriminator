# -*- coding: utf-8 -*-

from huggingface_hub import login
login(token='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', add_to_git_credential=True)

from datasets import load_dataset

dataset = load_dataset(r"E:\RealFakeDB_small")

dataset.push_to_hub("HDanh/RealFakeDB_small")


# =============================================================================
# # Assume index là vị trí của dòng bạn muốn cập nhật
# index_to_update = 0
# new_image_path = r"E:\RealFakeDB_small\train\real\000000479400.png"
# 
# # Cập nhật dữ liệu trong trường hợp thay đổi ảnh
# dataset["train"][index_to_update]["image"] = new_image_path
# 
# 
# =============================================================================
#E:\RealFakeDB_small\train\real\000000479400.png

#"E:\RealFakeDB_small\train\real\000000479400.png"