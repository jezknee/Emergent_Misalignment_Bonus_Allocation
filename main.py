import make_company
from allocation_prompts import build_prompt_table, save_prompt_table

table = build_prompt_table(
    50,
    variants=("assistant_fair", "analyst_fair"),
    sizes=[(6, 4), (10, 5), (20, 8), (30, 12), (50, 20)],
)
save_prompt_table(table, "prompts")