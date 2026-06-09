import re

H = 2056
names = {
    '03': '학생회관', '05': '수성관', '21': '정보통신대학', '22': '제1공학관',
    '23': '공과대학', '24': '공학실습동', '25': '제2공학관(25)', '26': '제2공학관(26)',
    '27': '제2공학관(27)', '31': '제1과학관', '32': '제2과학관', '33': '화학관',
    '40': '반도체관', '48': '삼성학술정보관', '51': '기초학문관', '53': '약학관',
    '61': '생명공학관(61)', '62': '생명공학관(62)', '70': '대강당', '71': '의학관',
    '83': '제2종합연구동', '85': '산학협력센터', '86': 'N센터'
}

with open('dis.txt', 'r', encoding='utf-8') as f:
    content = f.read()

matches = re.findall(r"'(\d+)': \((\d+), (\d+)\)", content)

print("BUILDINGS = {")
for bld, x, cy in matches:
    x = int(x)
    cy = int(cy)
    y = H - cy
    name = names.get(bld, f"건물 {bld}")
    print(f"    '{bld}': {{'name': '{name}', 'campus_x': {x}, 'campus_y': {y}}},")
print("}")
