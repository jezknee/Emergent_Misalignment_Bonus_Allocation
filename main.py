from make_company import export_companies
from allocation_prompts import build_prompt_table_from_folders, make_variants, save_prompt_table


export_companies(range(0, 10),  "companies", formats=("csv", "xlsx"), n_emp=3, n_proj=1)
export_companies(range(10, 45),  "companies", formats=("csv", "xlsx"), n_emp=5, n_proj=3)
export_companies(range(20, 45),  "companies", formats=("csv", "xlsx"), n_emp=10, n_proj=5)
export_companies(range(45, 60), "companies", formats=("csv", "xlsx"), n_emp=20, n_proj=8, support=True)
export_companies(range(60, 70), "companies", formats=("csv", "xlsx"), n_emp=30, n_proj=12, support=True)

table = build_prompt_table_from_folders(
    "companies",
    variants=make_variants(),   # 2 personas x 4 approval routes = 8 variants
    output_format="shares",
)
save_prompt_table(table, "prompts")
print(len(table), table["n_chars"].describe())

