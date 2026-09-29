from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .edges import (
    full_connect_edge_index,
    image_grid_edge_index,
    parse_patch_coord,
)
from .pack import concat_slide_graphs, event_from_censorship, pack_case
from .paths import HGCN_WSI_EXPERIMENT, hgcn_wsi_dir
from .rna import (
    HGCN_RNA_FAMILY_NAMES,
    family_vectors_from_tpm,
    gsea_contribution_vector,
    load_gene_families,
    parse_gmt_families,
    parse_html_families,
    ranked_gene_list,
    running_enrichment_score,
)


class NativeHgcnGraphTests(unittest.TestCase):
    def test_parse_row_col_and_hyphen(self):
        self.assertEqual(parse_patch_coord("23_25.png"), (23, 25))
        self.assertEqual(parse_patch_coord("23-25"), (23, 25))
        self.assertEqual(parse_patch_coord("0-11_7"), (11, 7))
        self.assertEqual(
            parse_patch_coord("TCGA-B0-4823-01Z-00-DX1.uuid/23-25.png"),
            (23, 25),
        )

    def test_image_eight_neighborhood(self):
        edge_index = image_grid_edge_index(["0_0", "0_1", "1_0"])
        pairs = {(int(s), int(e)) for s, e in edge_index.t().tolist()}
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertIn((0, 2), pairs)

    def test_image_edges_stay_inside_slide_prefix(self):
        edge_index = image_grid_edge_index(
            ["slideA/0_0", "slideA/0_1", "slideB/0_0", "slideB/0_1"]
        )
        pairs = {(int(s), int(e)) for s, e in edge_index.t().tolist()}
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 0), pairs)
        self.assertIn((2, 3), pairs)
        self.assertIn((3, 2), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((1, 3), pairs)
        self.assertNotIn((1, 1), pairs)

    def test_rna_cli_full_connect_no_self_loop(self):
        edge_index = full_connect_edge_index(5)
        self.assertEqual(tuple(edge_index.shape), (2, 20))
        self.assertTrue(all(s != e for s, e in edge_index.t().tolist()))

    def test_event_flips_tcga_censorship(self):
        self.assertEqual(event_from_censorship(1), 0)
        self.assertEqual(event_from_censorship(0), 1)

    def test_missing_modality_stays_out_of_data_type(self):
        data = pack_case(
            "TCGA-XX-0001",
            sur_type=event_from_censorship(1),
            survival_months=12.0,
            x_cli=torch.zeros((4, 1024)),
            edge_index_cli=full_connect_edge_index(4),
        )
        self.assertEqual(data.data_type, ["cli"])
        self.assertEqual(int(data.sur_type.item()), 0)
        self.assertEqual(int(data.x_img.shape[0]), 0)
        self.assertEqual(int(data.x_rna.shape[0]), 0)
        self.assertEqual(int(data.x_cli.shape[1]), 1024)

    def test_multi_slide_edges_stay_disjoint(self):
        x0 = torch.ones((2, 1024))
        x1 = torch.ones((2, 1024))
        e0 = torch.tensor([[0], [1]], dtype=torch.long)
        e1 = torch.tensor([[0], [1]], dtype=torch.long)
        x_img, edge_index = concat_slide_graphs([(x0, e0), (x1, e1)])
        self.assertEqual(int(x_img.shape[0]), 4)
        pairs = {(int(s), int(e)) for s, e in edge_index.t().tolist()}
        self.assertEqual(pairs, {(0, 1), (2, 3)})



class NativeHgcnWsiLayoutTests(unittest.TestCase):
    def test_wsi_graphs_land_under_hgcn_data_p(self):
        dest = hgcn_wsi_dir("tcga_kirc", repo_root=Path("/tmp/repo"))
        self.assertEqual(
            dest,
            Path("/tmp/repo") / "SurvPGC_Workspace" / "hgcn data" / "tcga_kirc" / "P" / HGCN_WSI_EXPERIMENT,
        )

    def test_merge_patient_patch_dict_keeps_all_slides(self):
        from .generate import merge_patient_patch_dict
        merged = merge_patient_patch_dict(
            [
                ("slideA.svs", {"23-25.png": np.ones(4, dtype=np.float32)}),
                ("slideB.svs", {"23-25.png": np.full(4, 2, dtype=np.float32)}),
            ]
        )
        self.assertEqual(len(merged), 2)
        self.assertIn("slideA/23-25.png", merged)
        self.assertIn("slideB/23-25.png", merged)
        from .edges import image_grid_edge_index

        edge_index = image_grid_edge_index(list(merged))
        self.assertEqual(int(edge_index.shape[1]), 0)


class NativeHgcnRnaGseaTests(unittest.TestCase):
    def test_parse_gmt_keeps_official_named_families(self):
        text = "\n".join(
            [
                "Tumor_Suppressor_Genes\thttp://example\tTS1\tTS2",
                "Oncogenes\thttp://example\tONC1",
                "Protein Kinases\thttp://example\tPK1",
                "Cell Differentiation Markers\thttp://example\tCDM1",
                "Translocated Cancer Genes\thttp://example\tTCG1",
                "Cytokines and Growth Factors\thttp://example\tCYT1",
                "Homeodomain Proteins\thttp://example\tHD1",
                "Transcription Factors\thttp://example\tTF1",
                "Hallmark_Apoptosis\thttp://example\tBAD",
            ]
        )
        families = parse_gmt_families(text)
        self.assertEqual(set(families), set(HGCN_RNA_FAMILY_NAMES))
        self.assertEqual(families["Tumor Suppressor Genes"], ["TS1", "TS2"])
        self.assertNotIn("Hallmark_Apoptosis", families)

    def test_parse_html_families_reads_official_page_columns(self):
        html = """
        </table>">tumor suppressors</a>
        <form name="columnHeading7" action="msigdb/human/annotate.jsp" method="POST">
            <input type="hidden" name="geneList" value="APC,TP53"/>
        </form>
        </table>">oncogenes</a>
        <form name="columnHeading6" action="msigdb/human/annotate.jsp" method="POST">
            <input type="hidden" name="geneList" value="MYC,KRAS"/>
        </form>
        """
        families = parse_html_families(html)
        self.assertEqual(families["Tumor Suppressor Genes"], ["APC", "TP53"])
        self.assertEqual(families["Oncogenes"], ["MYC", "KRAS"])

    def test_load_gene_families_from_gmt(self):
        text = "\n".join(
            f"{name.replace(' ', '_')}\thttp://example\t{name[:2].upper()}1\t{name[:2].upper()}2"
            for name in HGCN_RNA_FAMILY_NAMES
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "families.gmt"
            path.write_text(text, encoding="utf-8")
            families = load_gene_families(gmt_path=path)
        self.assertEqual(len(families), 8)
        self.assertEqual(families[0], ["TU1", "TU2"])

    def test_gsea_contribution_uses_hit_running_score(self):
        ranked = ["G1", "G2", "G3", "G4"]
        family = ["G1", "G3"]
        scores = running_enrichment_score(ranked, set(family))
        vector = gsea_contribution_vector(ranked, family, dim=4)
        self.assertTrue(np.allclose(vector[:2], [scores[0], scores[2]]))
        self.assertTrue(np.allclose(vector[2:], 0.0))

    def test_family_vectors_rank_by_tpm_then_pad(self):
        tpm = {"B": 10.0, "A": 5.0, "C": 1.0}
        families = [["A", "C"], ["B"]]
        matrix = family_vectors_from_tpm(tpm, families, dim=4)
        self.assertEqual(matrix.shape, (2, 4))
        ranked = ranked_gene_list(tpm)
        self.assertEqual(ranked, ["B", "A", "C"])
        first = gsea_contribution_vector(ranked, families[0], dim=4)
        self.assertTrue(np.allclose(matrix[0], first))



class NativeHgcnPklAssemblyTests(unittest.TestCase):
    def test_missing_wsi_and_rna_keep_clinic_only(self):
        import sys
        from pathlib import Path as _Path

        baseline = _Path(__file__).resolve().parents[1]
        if str(baseline) not in sys.path:
            sys.path.insert(0, str(baseline))
        import hgcn_graph_build
        import joblib
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = _Path(tmp)
            wsi_root = tmp_path / "P" / "kimianet"
            gene_root = tmp_path / "G" / "msigdb_gsea_families"
            clinic_root = tmp_path / "C" / "L0"
            wsi_root.mkdir(parents=True)
            gene_root.mkdir(parents=True)
            clinic_root.mkdir(parents=True)
            joblib.dump({}, wsi_root / "t_img_fea.pkl")
            joblib.dump({}, gene_root / "t_rna_fea.pkl")
            x_cli = {"TCGA-XX-0001": np.zeros((4, 1024), dtype=np.float32)}
            edge = np.array([[0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 2, 3]], dtype=np.int64)
            joblib.dump(x_cli, clinic_root / "x_cli.pkl")
            joblib.dump(edge, clinic_root / "edge_index_cli.pkl")
            joblib.dump({"TCGA-XX-0001": [None, None, None, None]}, clinic_root / "t_cli_feas.pkl")
            joblib.dump({"TCGA-XX-0001": [None, None, None, None]}, clinic_root / "ttt_cli_feas.pkl")
            case_df = pd.DataFrame(
                [{"case_id": "TCGA-XX-0001", "slide_id": "slideA.svs", "censorship": 1, "survival_months": 12.0}]
            )
            data = hgcn_graph_build.assemble_case_graph(
                "TCGA-XX-0001",
                case_df,
                wsi_root=wsi_root,
                gene_root=gene_root,
                clinic_root=clinic_root,
            )
            self.assertEqual(data.data_type, ["cli"])
            self.assertEqual(int(data.x_img.shape[0]), 0)
            self.assertEqual(int(data.x_rna.shape[0]), 0)
            self.assertEqual(tuple(data.x_cli.shape), (4, 1024))

    def test_wsi_pkl_builds_eight_neighborhood(self):
        import sys
        from pathlib import Path as _Path

        baseline = _Path(__file__).resolve().parents[1]
        if str(baseline) not in sys.path:
            sys.path.insert(0, str(baseline))
        import hgcn_graph_build

        patches = {
            "slideA/0_0.png": np.ones(1024, dtype=np.float32),
            "slideA/0_1.png": np.full(1024, 2, dtype=np.float32),
            "slideB/0_0.png": np.full(1024, 3, dtype=np.float32),
        }
        graph = hgcn_graph_build.graph_from_patch_dict(patches)
        self.assertEqual(tuple(graph["x"].shape), (3, 1024))
        pairs = {(int(s), int(e)) for s, e in graph["edge_index"].t().tolist()}
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_assemble_uses_event_not_censorship(self):
        import sys
        from pathlib import Path as _Path

        baseline = _Path(__file__).resolve().parents[1]
        if str(baseline) not in sys.path:
            sys.path.insert(0, str(baseline))
        import hgcn_graph_build
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = _Path(tmp)
            wsi_root = tmp_path / "P" / "kimianet"
            gene_root = tmp_path / "G" / "msigdb_gsea_families"
            clinic_root = tmp_path / "C" / "L0"
            wsi_root.mkdir(parents=True)
            gene_root.mkdir(parents=True)
            clinic_root.mkdir(parents=True)
            import joblib
            joblib.dump({}, wsi_root / "t_img_fea.pkl")
            joblib.dump({}, gene_root / "t_rna_fea.pkl")
            x_cli = {"TCGA-XX-0001": np.zeros((4, 1024), dtype=np.float32)}
            edge = np.array([[0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 2, 3]], dtype=np.int64)
            joblib.dump(x_cli, clinic_root / "x_cli.pkl")
            joblib.dump(edge, clinic_root / "edge_index_cli.pkl")
            joblib.dump({"TCGA-XX-0001": [None, None, None, None]}, clinic_root / "t_cli_feas.pkl")
            joblib.dump({"TCGA-XX-0001": [None, None, None, None]}, clinic_root / "ttt_cli_feas.pkl")
            case_df = pd.DataFrame(
                [{"case_id": "TCGA-XX-0001", "slide_id": "slideA.svs", "censorship": 1, "survival_months": 12.0}]
            )
            data = hgcn_graph_build.assemble_case_graph(
                "TCGA-XX-0001",
                case_df,
                wsi_root=wsi_root,
                gene_root=gene_root,
                clinic_root=clinic_root,
            )
            self.assertEqual(int(data.sur_type.item()), 0)

    def test_empty_requested_modality_forward_does_not_crash(self):
        from types import SimpleNamespace
        import importlib.util
        import sys

        hgcn_dir = Path(__file__).resolve().parents[1] / "third_party" / "HGCN" / "HGCN_code"
        mae_path = hgcn_dir / "mae_model.py"
        sys.path.insert(0, str(hgcn_dir))
        try:
            spec = importlib.util.spec_from_file_location("survpgc_test_mae_model", mae_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if sys.path and sys.path[0] == str(hgcn_dir):
                sys.path.pop(0)
        model = module.fusion_model_mae_2(
            img_in_feats=1024,
            rna_in_feats=1024,
            cli_in_feats=1024,
            n_hidden=512,
            out_classes=512,
            dropout=0.0,
            train_type_num=3,
        )
        model.eval()
        graph = SimpleNamespace(
            x_img=torch.zeros((0, 1024)),
            x_rna=torch.zeros((1, 1024)),
            x_cli=torch.zeros((2, 1024)),
            data_id="TCGA-XX-0001",
            edge_index_image=torch.zeros((2, 0), dtype=torch.long),
            edge_index_rna=torch.zeros((2, 0), dtype=torch.long),
            edge_index_cli=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        with torch.no_grad():
            (one_x, multi_x), _, _, fea_dict = model(
                graph,
                train_use_type=["img", "rna", "cli"],
                use_type=["img", "rna", "cli"],
                in_mask=[[[True, False, False]]],
                mix=False,
            )
        self.assertEqual(tuple(one_x.shape), (1,))
        self.assertEqual(tuple(multi_x.shape), (3, 1))
        self.assertEqual(tuple(fea_dict["mae_labels"].shape), (3, 512))

    def test_split_intersection_keeps_assembled_ids_only(self):
        import importlib.util
        import sys

        hgcn_dir = Path(__file__).resolve().parents[1] / "third_party" / "HGCN" / "HGCN_code"
        train_path = hgcn_dir / "train.py"
        sys.path.insert(0, str(hgcn_dir))
        try:
            spec = importlib.util.spec_from_file_location("survpgc_test_hgcn_train", train_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if sys.path and sys.path[0] == str(hgcn_dir):
                sys.path.pop(0)
        kept = module._intersect_split_ids(
            ["keep-a", "drop-b", "keep-c"],
            ["keep-a", "keep-c"],
            "train",
            "splits_0.csv",
        )
        self.assertEqual(kept, ["keep-a", "keep-c"])
        with self.assertRaises(ValueError):
            module._intersect_split_ids(["drop-b"], ["keep-a"], "test", "splits_0.csv")


if __name__ == "__main__":
    unittest.main()
