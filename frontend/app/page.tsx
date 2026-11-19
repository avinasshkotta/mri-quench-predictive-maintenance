"use client";

import { useCallback, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import FleetPanel from "@/components/FleetPanel";
import MagnetDetail from "@/components/MagnetDetail";
import StatusBar from "@/components/StatusBar";
import type { StatusResponse } from "@/lib/types";

export default function Page() {
  const [systemStatus, setSystemStatus] = useState<StatusResponse | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [selectedMagnet, setSelectedMagnet] = useState<string | null>(null);
  const [detailRefreshKey, setDetailRefreshKey] = useState(0);

  const handleStatusChange = useCallback((status: StatusResponse | null, reachable: boolean) => {
    setSystemStatus(status);
    setApiOnline(reachable);
  }, []);

  const handleSelectMagnet = useCallback((magnetId: string) => {
    setSelectedMagnet(magnetId);
    setDetailRefreshKey((key) => key + 1);
  }, []);

  /** The chat agent investigates a magnet -> focus that magnet in the side panel. */
  const handleMagnetHint = useCallback((magnetId: string) => {
    setSelectedMagnet(magnetId);
    setDetailRefreshKey((key) => key + 1);
  }, []);

  return (
    <div className="shell">
      <StatusBar onStatusChange={handleStatusChange} />
      <main className="app">
        <div className="workspace">
          <ChatPanel
            llmConfigured={systemStatus ? systemStatus.llm_configured : null}
            apiOnline={apiOnline}
            model={systemStatus?.model ?? null}
            onMagnetHint={handleMagnetHint}
          />
          <aside className="side-column" aria-label="Fleet risk and magnet detail">
            <FleetPanel selectedId={selectedMagnet} onSelect={handleSelectMagnet} />
            <MagnetDetail magnetId={selectedMagnet} refreshKey={detailRefreshKey} />
          </aside>
        </div>
      </main>
    </div>
  );
}
