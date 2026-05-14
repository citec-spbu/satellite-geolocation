import numpy as np
import torch
from tqdm import tqdm

def extract_features(model, loader, is_drone_flag, device):
    feats, labels, heights, paths = [], [], [], []
    with torch.no_grad():
        for img, lbl, _, h, path in tqdm(loader, desc="Извлечение признаков"):
            img = img.to(device)
            dr_flag = torch.full((img.size(0),), is_drone_flag, device=device)
            with torch.amp.autocast('cuda'):
                feat = model(img, dr_flag)
            feats.append(feat.cpu())
            labels.append(lbl)
            heights.append(h)
            paths.extend(path)
    if len(feats) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), []
    return torch.cat(feats), torch.cat(labels), torch.cat(heights), paths

def calculate_metrics(query_feat, query_labels, gallery_feat, gallery_labels):
    dist_matrix = torch.matmul(query_feat, gallery_feat.T)
    indices = torch.argsort(dist_matrix, dim=1, descending=True)

    recall_1, recall_5, recall_10 = 0, 0, 0
    all_ap = []

    for i in range(len(query_labels)):
        label = query_labels[i]
        ranked_labels = gallery_labels[indices[i]]
        correct_pos = (ranked_labels == label).nonzero(as_tuple=False).flatten()
        if len(correct_pos) > 0:
            first_match = correct_pos[0].item()
            if first_match < 1: recall_1 += 1
            if first_match < 5: recall_5 += 1
            if first_match < 10: recall_10 += 1

            ap = 0.0
            for j, pos in enumerate(correct_pos):
                ap += (j + 1) / (pos.item() + 1)
            ap /= len(correct_pos)
            all_ap.append(ap)
        else:
            all_ap.append(0.0)

    num_queries = len(query_labels)
    return {
        "R@1": recall_1 / num_queries,
        "R@5": recall_5 / num_queries,
        "R@10": recall_10 / num_queries,
        "mAP": np.mean(all_ap)
    }

def evaluate_model(model, mode, root_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    query_dataset = SUESDataset(root_dir=root_data, transform=get_transforms(mode, is_train=False), split='test', view='drone')
    gallery_dataset = SUESDataset(root_dir=root_data, transform=get_transforms(mode, is_train=False), split='test', view='satellite')

    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=2)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=2)

    gallery_feat, gallery_labels, _, gallery_paths = extract_features(model, gallery_loader, False, device)
    query_feat, query_labels, query_heights, query_paths = extract_features(model, query_loader, True, device)

    if query_feat.numel() == 0 or gallery_feat.numel() == 0: return None, 0.0

    results = calculate_metrics(query_feat, query_labels, gallery_feat, gallery_labels)

    print("\n")
    print(f"mAP: {results['mAP']:.4f} | R@1: {results['R@1']:.4f} | R@5: {results['R@5']:.4f} | R@10: {results['R@10']:.4f}")

    unique_heights = [150, 200, 250, 300]
    for h_val in unique_heights:
        mask = (query_heights == h_val)
        if mask.any():
            h_feat = query_feat[mask]
            h_labels = query_labels[mask]
            h_res = calculate_metrics(h_feat, h_labels, gallery_feat, gallery_labels)
            print(f"Высота {h_val}м: mAP: {h_res['mAP']:.4f} | R@1: {h_res['R@1']:.4f} | R@5: {h_res['R@5']:.4f} | R@10: {h_res['R@10']:.4f}")

    return results, results['mAP']
