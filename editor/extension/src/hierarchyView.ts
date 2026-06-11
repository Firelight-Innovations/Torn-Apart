// Scene hierarchy tree (EDITOR_PRD F2) — the Unity-style Hierarchy dock.
// A native VS Code TreeView backed by a flat list of scene objects fetched from
// the daemon (scene.tree) and refreshed on every scene.changed notification.
// Tree elements are object ids (numbers); drag-and-drop reparents via the
// daemon. Selection is wired to the 3D viewport by extension.ts.
import * as vscode from "vscode";

export interface SceneObjectDTO {
  id: number;
  name: string;
  kind: string;
  parent: number | null;
  position: number[];
  rotation: number[];
  scale: number[];
}

const DND_MIME = "application/vnd.fire-editor.sceneobject";

const KIND_ICON: Record<string, string> = {
  empty: "symbol-namespace",
  cube: "symbol-constant",
  sphere: "circle-large-outline",
  light: "lightbulb",
  spawn: "person",
};

export class HierarchyProvider
  implements vscode.TreeDataProvider<number>, vscode.TreeDragAndDropController<number>
{
  readonly dropMimeTypes = [DND_MIME];
  readonly dragMimeTypes = [DND_MIME];

  private readonly _onDidChange = new vscode.EventEmitter<number | undefined | void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private objects = new Map<number, SceneObjectDTO>();
  private childIndex = new Map<number | null, number[]>();

  /** Called when a drag-drop asks to reparent `id` under `parent` (null = root). */
  onReparent: (id: number, parent: number | null) => void = () => {};

  /** Replace the whole tree from a fresh scene.tree / scene.changed payload. */
  setObjects(list: SceneObjectDTO[]): void {
    this.objects = new Map(list.map((o) => [o.id, o]));
    this.childIndex = new Map();
    for (const o of list) {
      const key = o.parent ?? null;
      const arr = this.childIndex.get(key) ?? [];
      arr.push(o.id);
      this.childIndex.set(key, arr);
    }
    this._onDidChange.fire();
  }

  get(id: number): SceneObjectDTO | undefined {
    return this.objects.get(id);
  }

  has(id: number): boolean {
    return this.objects.has(id);
  }

  // --- TreeDataProvider ---
  getChildren(element?: number): number[] {
    return this.childIndex.get(element ?? null) ?? [];
  }

  getParent(element: number): number | undefined {
    return this.objects.get(element)?.parent ?? undefined;
  }

  getTreeItem(id: number): vscode.TreeItem {
    const obj = this.objects.get(id);
    const hasChildren = (this.childIndex.get(id) ?? []).length > 0;
    const item = new vscode.TreeItem(
      obj?.name ?? `#${id}`,
      hasChildren
        ? vscode.TreeItemCollapsibleState.Expanded
        : vscode.TreeItemCollapsibleState.None
    );
    item.id = String(id);
    item.contextValue = "sceneObject";
    item.iconPath = new vscode.ThemeIcon(KIND_ICON[obj?.kind ?? "empty"] ?? "circle-outline");
    item.description = obj?.kind;
    item.tooltip = obj ? `${obj.name} (${obj.kind})  id ${obj.id}` : undefined;
    return item;
  }

  // --- Drag & drop ---
  handleDrag(source: readonly number[], data: vscode.DataTransfer): void {
    data.set(DND_MIME, new vscode.DataTransferItem(source[0]));
  }

  handleDrop(target: number | undefined, data: vscode.DataTransfer): void {
    const item = data.get(DND_MIME);
    if (!item) return;
    const dragged = Number(item.value);
    if (!Number.isFinite(dragged) || dragged === target) return;
    // Dropping onto empty space (no target) promotes to a root.
    this.onReparent(dragged, target ?? null);
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
