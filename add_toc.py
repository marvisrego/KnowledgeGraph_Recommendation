"""
Add a Table of Contents to Career_KG_Submission.docx
matching the style of Sample_proposal.docx.
"""
import copy
import shutil
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import lxml.etree as etree

SAMPLE = r'C:\Users\marvi\OneDrive\Documents\MDX\THESIS\Thesis_VScode\Proposal\New folder\Sample_proposal.docx'
TARGET = r'C:\Users\marvi\OneDrive\Documents\MDX\THESIS\Thesis_VScode\Proposal\Final\Career_KG_Submission.docx'
BACKUP = r'C:\Users\marvi\OneDrive\Documents\MDX\THESIS\Thesis_VScode\Proposal\Final\Career_KG_Submission_backup.docx'
OUTPUT = r'C:\Users\marvi\OneDrive\Documents\MDX\THESIS\Thesis_VScode\Proposal\Final\Career_KG_Submission.docx'

# Always work from the backup so re-runs don't double-insert
import os
if os.path.exists(BACKUP):
    shutil.copy2(BACKUP, TARGET)
    print("Restored from backup.")
else:
    shutil.copy2(TARGET, BACKUP)
    print("Backup created.")

# ── 1. Load both documents ──────────────────────────────────────────────────
sample_doc = Document(SAMPLE)
target_doc = Document(TARGET)

# ── 2. Copy TOC styles from sample into target ────────────────────────────
sample_styles_xml = sample_doc.styles.element
target_styles_xml = target_doc.styles.element

TOC_STYLE_IDS = ('TOC1', 'TOC2', 'TOC3')

def style_exists(styles_el, style_id):
    for s in styles_el.findall(qn('w:style')):
        val = s.get(qn('w:styleId'))
        if val == style_id:
            return True
    return False

copied = []
for s in sample_styles_xml.findall(qn('w:style')):
    sid = s.get(qn('w:styleId'))
    if sid in TOC_STYLE_IDS:
        if not style_exists(target_styles_xml, sid):
            target_styles_xml.append(copy.deepcopy(s))
            copied.append(sid)
        else:
            # Replace existing with sample version
            for old in target_styles_xml.findall(qn('w:style')):
                if old.get(qn('w:styleId')) == sid:
                    target_styles_xml.remove(old)
                    break
            target_styles_xml.append(copy.deepcopy(s))
            copied.append(f'{sid}(replaced)')

print(f"Styles copied/updated: {copied}")

# ── 3. Build TOC paragraphs ────────────────────────────────────────────────
# Each entry: (style_id, label_text)
# style_id: 'TOC1' = top level, 'TOC2' = subsection, 'TOC3' = sub-subsection
# We use a tab (\t) to separate number/title from page (right-aligned).
# Page numbers are left as blank so the user can fill them in or use Word's
# "Update Table" after assigning Heading styles to section titles.

TOC_ENTRIES = [
    # (style_id, text)
    # ── Proposal ──
    ('TOC1', '1.\tAbstract'),
    ('TOC1', '2.\tIntroduction'),
    ('TOC1', '3.\tState of the Art Review'),
    ('TOC1', '4.\tMethods'),
    ('TOC1', '5.\tProposed Work'),
    ('TOC2', '5.1.\tIntended contribution of the work'),
    ('TOC2', '5.2.\tBenefits of the proposed work'),
    ('TOC2', '5.3.\tProcedures and activities of the proposed work'),
    ('TOC2', '5.4.\tEvaluation of the project'),
    ('TOC2', '5.5.\tResources needed'),
    ('TOC2', '5.6.\tAccess to participants and clients'),
    ('TOC2', '5.7.\tEthical Aspects'),
    ('TOC1', '6.\tWork Plan and Schedule'),
    ('TOC1', '7.\tConclusions'),
    ('TOC1', '8.\tAppendixes'),
    # ── Literature Review ──
    # Chapter 2 title is written as a single label (no internal tab) so it
    # doesn't split across the number/title columns like "Chapter 2: | title"
    ('TOC1', 'Chapter 2: Literature Review — Knowledge Graphs, Career Recommendation & Explainability'),
    ('TOC2', '2.1.\tPurpose and Scope of the Review'),
    ('TOC2', '2.2.\tReview Methodology and Corpus Construction'),
    ('TOC3', '2.2.1.\tSearch Strategy'),
    ('TOC3', '2.2.2.\tInclusion and Exclusion Criteria'),
    ('TOC3', '2.2.3.\tCorpus Composition'),
    ('TOC2', '2.3.\tEvolution of Career Recommendation Research'),
    ('TOC2', '2.4.\tComposition of the Reviewed Literature'),
    ('TOC2', '2.5.\tCore Thematic Review'),
    ('TOC3', '2.5.1.\tKnowledge Graph Architectures for Recommendation'),
    ('TOC3', '2.5.2.\tCareer-Oriented Recommendation and Explainability'),
    ('TOC3', '2.5.3.\tSkill Representation, Title Normalisation, and Labour-Market Data'),
    ('TOC3', '2.5.4.\tLearning-Path and Course Recommendation Systems'),
    ('TOC2', '2.6.\tOutput of the Review: Towards an Integrated Career Knowledge Graph'),
    ('TOC2', '2.7.\tSummary and Research Gap'),
    ('TOC1', 'Bibliography'),
]

def make_toc_paragraph(doc, style_id, text):
    """Create a paragraph with the given TOC style and text."""
    p = OxmlElement('w:p')

    pPr = OxmlElement('w:pPr')
    pStyle = OxmlElement('w:pStyle')
    pStyle.set(qn('w:val'), style_id)
    pPr.append(pStyle)
    p.append(pPr)

    # Split by tab character to separate number/title from page
    parts = text.split('\t')
    for idx, part in enumerate(parts):
        r = OxmlElement('w:r')
        t = OxmlElement('w:t')
        t.text = part
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        r.append(t)
        p.append(r)
        # Add tab between parts (not after last)
        if idx < len(parts) - 1:
            r_tab = OxmlElement('w:r')
            tab = OxmlElement('w:tab')
            r_tab.append(tab)
            p.append(r_tab)

    return p

def make_heading_paragraph(doc, text, style_name='Heading 1'):
    """Create a heading paragraph."""
    p_obj = doc.add_paragraph(text, style=style_name)
    return p_obj._element

# ── 4. Find insertion point: just before "1. Abstract" paragraph ────────────
# That is paragraph index 31 in the target doc (0-indexed)
insert_before_para_text = '1. Abstract'
insert_before_idx = None
for i, p in enumerate(target_doc.paragraphs):
    if p.text.strip() == insert_before_para_text:
        insert_before_idx = i
        break

if insert_before_idx is None:
    raise ValueError(f"Could not find paragraph '{insert_before_para_text}'")

print(f"Inserting TOC before paragraph index {insert_before_idx}: {target_doc.paragraphs[insert_before_idx].text[:50]!r}")

# ── 5. Build the list of elements to insert ────────────────────────────────
# We'll insert them BEFORE the paragraph at insert_before_idx

# Reference element to insert before
ref_para_el = target_doc.paragraphs[insert_before_idx]._element
body = ref_para_el.getparent()

# Collect elements in order (they will be inserted in reverse to maintain order)
new_elements = []

# (a) Page break paragraph before TOC heading
page_break_p = OxmlElement('w:p')
pb_pPr = OxmlElement('w:pPr')
pb_pStyle = OxmlElement('w:pStyle')
pb_pStyle.set(qn('w:val'), 'Heading1')
pb_pPr.append(pb_pStyle)
page_break_p.append(pb_pPr)
pb_r = OxmlElement('w:r')
pb_br = OxmlElement('w:br')
pb_br.set(qn('w:type'), 'page')
pb_r.append(pb_br)
page_break_p.append(pb_r)
new_elements.append(page_break_p)

# (b) "Table of Contents" heading
toc_heading = OxmlElement('w:p')
th_pPr = OxmlElement('w:pPr')
th_pStyle = OxmlElement('w:pStyle')
th_pStyle.set(qn('w:val'), 'Heading1')
th_pPr.append(th_pStyle)
toc_heading.append(th_pPr)
th_r = OxmlElement('w:r')
th_t = OxmlElement('w:t')
th_t.text = 'Table of Contents'
th_r.append(th_t)
toc_heading.append(th_r)
new_elements.append(toc_heading)

# (c) TOC entries
for style_id, text in TOC_ENTRIES:
    new_elements.append(make_toc_paragraph(target_doc, style_id, text))

# (d) Empty paragraph after TOC (spacer before content)
spacer = OxmlElement('w:p')
new_elements.append(spacer)

# ── 6. Insert all elements before the reference paragraph ─────────────────
for el in new_elements:
    body.insert(list(body).index(ref_para_el), el)

print(f"Inserted {len(new_elements)} elements (1 page break + 1 heading + {len(TOC_ENTRIES)} TOC entries + 1 spacer).")

# ── 7. Save ────────────────────────────────────────────────────────────────
target_doc.save(OUTPUT)
print(f"Saved: {OUTPUT}")
print("Done! Open the document in Word and the TOC is ready.")
print("Note: Page numbers are not included since the document uses plain paragraph")
print("styles. To add page numbers, apply Heading 1/2/3 styles to section titles")
print("in Word, then right-click the TOC and choose 'Update Field'.")
